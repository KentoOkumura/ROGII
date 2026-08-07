# %% [markdown]
# # exp408 HMM message / rate basin audit — train
#
# Re-run the unchanged exp209 exact HMM on the preregistered 450 wells and
# observe its predictive, filtered, and smoothed sufficient statistics.  This
# notebook creates no prediction candidate: the regenerated posterior mean is
# used only for strict exp270 parity.

# %% [markdown]
# ## Contents
# 1. Imports and fixed contract
# 2. Notebook-safe paths, SHA, and leakage guards
# 3. Target-free input loading
# 4. Exact exp209 HMM input preparation
# 5. Exact forward-backward with message sufficient statistics
# 6. Target-free message freeze and transition moments
# 7. Truth-late basin and hidden-rate readout
# 8. Episode attribution
# 9. Kaggle CPU orchestration
# 10. Metrics and generated artifacts

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from numba import njit, prange, set_num_threads

EXPERIMENT_NAME = "exp408_hmm_message_rate_basin_audit"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
NEGATIVE_LOG_SENTINEL = np.float32(-1.0e18)

TARGET_FREE_MODE_COLUMNS = (
    "id",
    "well",
    "row_idx",
    "posterior_mean",
    "marginal_map",
    "topk_path_1",
    "md_since",
)
FORBIDDEN_DECODER_COLUMNS = {
    "TVT",
    "tvt_true",
    "true_tvt_readout_only",
    "error",
    "abs_error",
    "episode_id",
    "start_row_idx",
    "end_row_idx_exclusive",
}


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def validate_execution_contract(config: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "active_hmm_variants": int(get_nested(config, "execution.active_hmm_variants")),
        "hmm_well_runs": int(get_nested(config, "execution.total_hmm_well_runs")),
        "lightgbm_configs": int(get_nested(config, "execution.lightgbm_configs")),
        "trained_folds": int(get_nested(config, "execution.trained_folds")),
        "boosters": int(get_nested(config, "execution.boosters")),
        "models": int(get_nested(config, "execution.models")),
        "pf_well_runs": int(get_nested(config, "execution.pf_well_runs")),
        "beam_well_runs": int(get_nested(config, "execution.beam_well_runs")),
        "gpu_runs": int(get_nested(config, "execution.gpu_runs")),
    }
    expected = {
        "active_hmm_variants": 1,
        "hmm_well_runs": 450,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "models": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    if counts != expected:
        raise RuntimeError(f"execution contract changed: {counts} != {expected}")
    if not bool(get_nested(config, "execution.parent_control_regeneration")):
        raise RuntimeError("current parent HMM regeneration must be explicit")
    if not bool(get_nested(config, "execution.kaggle_execution_approved")):
        raise RuntimeError("Kaggle execution is not approved")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
        raise RuntimeError("exp408 is CPU-only")
    if bool(get_nested(config, "inference.enabled")):
        raise RuntimeError("inference must remain disabled")
    return counts


# %% [markdown]
# ## 2. Notebook-safe paths, SHA, and leakage guards

# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp408 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    resolved = config_path() if path is None else path
    value = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping")
    return value


def output_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        target = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        target = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    target.mkdir(parents=True, exist_ok=True)
    return target


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
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def normalized_frame(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    selected = frame if columns is None else frame.loc[:, list(columns)]
    result = selected.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].astype(np.float64)
        elif pd.api.types.is_integer_dtype(result[column]):
            result[column] = result[column].astype(np.int64)
        else:
            result[column] = result[column].astype(str)
    return result


def logical_frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    normalized = normalized_frame(frame, columns)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "logical_sha256": logical_frame_sha256(frame),
        "rows": len(frame),
        "columns": len(frame.columns),
    }


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0**3)
    return value / (1024.0**2)


def runtime_versions() -> dict[str, str]:
    import numba

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
    }


def resolve_bootstrap_asset(filename: str, local_path: str) -> Path:
    candidates = (
        Path.cwd() / "assets" / filename,
        Path.cwd() / filename,
        find_project_root() / local_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"bootstrap asset not found: {filename}")


def resolve_unique_file(
    *,
    filename: str,
    candidates: Iterable[str],
    patterns: Iterable[str] = (),
) -> Path:
    root = find_project_root()
    checked: list[str] = []
    matches: list[Path] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.is_file() and path.name == filename:
                matches.append(path)
            elif path.is_dir():
                direct = path / filename
                if direct.is_file():
                    matches.append(direct)
                matches.extend(sorted(path.rglob(filename)))
    if KAGGLE_INPUT_ROOT.is_dir():
        matches.extend(sorted(KAGGLE_INPUT_ROOT.rglob(filename)))
        for pattern in patterns:
            matches.extend(sorted(KAGGLE_INPUT_ROOT.glob(str(pattern))))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    if not unique:
        raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")
    if len(unique) > 1:
        hashes = {sha256_file(path): path for path in unique}
        if len(hashes) != 1:
            raise RuntimeError(f"ambiguous non-identical inputs for {filename}: {unique}")
    return unique[0]


def require_sha(path: Path, expected: str, *, decompressed: bool = False) -> str:
    actual = sha256_decompressed_csv(path) if decompressed else sha256_file(path)
    if expected and actual != expected:
        raise RuntimeError(
            f"SHA mismatch for {path}: expected={expected}, actual={actual}"
        )
    return actual


@dataclass
class LeakageLedger:
    frozen_wells: set[str] = field(default_factory=set)
    target_well_scope_reads: int = 0
    target_free_rows: int = 0
    truth_rows_before_well_freeze: int = 0
    episode_rows_before_well_freeze: int = 0
    truth_rows_after_well_freeze: int = 0
    episode_rows_after_well_freeze: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def record_scope(self, rows: int) -> None:
        self.target_well_scope_reads += int(rows)
        self.events.append({"phase": "scope_only", "rows": int(rows)})

    def record_target_free(self, label: str, rows: int) -> None:
        self.target_free_rows += int(rows)
        self.events.append({"phase": "target_free", "label": label, "rows": int(rows)})

    def freeze(self, well: str, prediction_sha: str, message_sha: str) -> None:
        self.frozen_wells.add(str(well))
        self.events.append(
            {
                "phase": "well_freeze",
                "well": str(well),
                "prediction_sha256": prediction_sha,
                "message_sha256": message_sha,
            }
        )

    def record_truth_late(self, well: str, rows: int) -> None:
        if str(well) not in self.frozen_wells:
            self.truth_rows_before_well_freeze += int(rows)
            raise RuntimeError(f"{well}: truth read before well message freeze")
        self.truth_rows_after_well_freeze += int(rows)
        self.events.append(
            {"phase": "truth_late", "well": str(well), "rows": int(rows)}
        )

    def record_episode_late(self, well: str, rows: int) -> None:
        if str(well) not in self.frozen_wells:
            self.episode_rows_before_well_freeze += int(rows)
            raise RuntimeError(f"{well}: episode details read before well message freeze")
        self.episode_rows_after_well_freeze += int(rows)
        self.events.append(
            {"phase": "episode_late", "well": str(well), "rows": int(rows)}
        )


# %% [markdown]
# ## 3. Target-free input loading

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
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        first = next(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"), None)
        if first is not None:
            return first.parent
    return find_project_root() / str(get_nested(config, "data.train_dir"))


def load_target_wells(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[list[str], dict[str, Any]]:
    spec = get_nested(config, "data.target_wells")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    sha = require_sha(path, str(spec["expected_sha256"]))
    frame = pd.read_csv(path, dtype={"well": str})
    wells = sorted(frame["well"].astype(str).unique())
    expected = int(get_nested(config, "validation.expected_target_wells"))
    if len(frame) != expected or len(wells) != expected:
        raise RuntimeError(f"target wells={len(wells)}/{expected}")
    ledger.record_scope(len(wells))
    return wells, {"path": str(path), "sha256": sha, "rows": len(frame)}


def load_target_free_mode_bank(
    config: Mapping[str, Any],
    target_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp270_mode_bank")
    path = resolve_unique_file(
        filename=str(spec["aggregate_filename"]),
        candidates=[str(value) for value in spec["aggregate_candidates"]],
        patterns=[str(value) for value in spec["aggregate_patterns"]],
    )
    decompressed_sha = require_sha(
        path,
        str(spec["expected_aggregate_decompressed_sha256"]),
        decompressed=True,
    )
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=list(TARGET_FREE_MODE_COLUMNS),
        dtype={"id": str, "well": str},
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk["well"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True)
    overlap = FORBIDDEN_DECODER_COLUMNS.intersection(frame.columns)
    if overlap:
        raise RuntimeError(f"target-free mode bank contains forbidden columns: {overlap}")
    frame["well"] = frame["well"].astype(str)
    frame["id"] = frame["id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well", "row_idx"]).any():
        raise RuntimeError("exp270 target-free keys are not unique")
    expected_rows = int(get_nested(config, "validation.expected_suffix_rows"))
    if len(frame) != expected_rows:
        raise RuntimeError(f"target-free rows={len(frame)}/{expected_rows}")
    if frame["well"].nunique() != len(target_wells):
        raise RuntimeError("target-free mode bank is missing target wells")
    ledger.record_target_free("exp270_mode_bank", len(frame))
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
    }


def load_target_free_folds(
    config: Mapping[str, Any],
    target_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_fold_source")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(value) for value in spec["candidates"]],
        patterns=[str(value) for value in spec["patterns"]],
    )
    decompressed_sha = require_sha(
        path,
        str(spec["expected_decompressed_sha256"]),
        decompressed=True,
    )
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["well_id", "row_idx", "fold"],
        dtype={"well_id": str},
        chunksize=250_000,
    ):
        selected = chunk.loc[chunk["well_id"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True).rename(columns={"well_id": "well"})
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int8)
    if frame.duplicated(["well", "row_idx"]).any():
        raise RuntimeError("fold keys are not unique")
    expected_rows = int(get_nested(config, "validation.expected_suffix_rows"))
    if len(frame) != expected_rows:
        raise RuntimeError(f"fold rows={len(frame)}/{expected_rows}")
    ledger.record_target_free("exp226_fold_identity", len(frame))
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
    }


def attach_target_free_folds(mode_bank: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    joined = mode_bank.merge(
        folds,
        on=["well", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if joined["fold"].isna().any():
        raise RuntimeError("missing fold after target-free identity join")
    return joined.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)


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
    forbidden = FORBIDDEN_DECODER_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise RuntimeError(f"{well}: target-free raw contains forbidden columns {forbidden}")
    typewell = pd.read_csv(typewell_path).sort_values("TVT").reset_index(drop=True)
    ledger.record_target_free(f"raw_horizontal:{well}", len(horizontal))
    ledger.record_target_free(f"raw_typewell:{well}", len(typewell))
    return horizontal, typewell


# %% [markdown]
# ## 4. Exact exp209 HMM input preparation

# %%
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
    tail_n: int = 30,
) -> tuple[float, float, float, float, int, int]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_gr = known["GR"].to_numpy(np.float64)
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    valid = np.isfinite(known_gr) & np.isfinite(typewell_at_known)
    if valid.sum() >= 20 and np.std(typewell_at_known[valid]) > 1.0e-6:
        cal_a, cal_b = np.polyfit(typewell_at_known[valid], known_gr[valid], 1)
    elif valid.any():
        cal_a = 1.0
        cal_b = float(np.nanmean(known_gr) - np.nanmean(typewell_at_known))
    else:
        cal_a, cal_b = 1.0, 0.0
    residual = known_gr[valid] - (cal_a * typewell_at_known[valid] + cal_b)
    if valid.sum() > 20:
        sigma = float(
            np.clip(
                1.4826 * np.median(np.abs(residual - np.median(residual))),
                8.0,
                60.0,
            )
        )
    else:
        sigma = 30.0
    init_rate, effective_rows, valid_steps = robust_initial_rate(known, tail_n)
    return (
        float(cal_a),
        float(cal_b),
        sigma,
        init_rate,
        effective_rows,
        valid_steps,
    )


def fixed_hmm_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    hmm = dict(get_nested(config, "model.hmm"))
    expected = {
        "step": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "df": 4.0,
        "emission": "gauss",
        "lam": 1.0,
        "sigma_mode": "std",
        "start_sig": 0.75,
        "r0_sig": 0.01,
        "band_pad": 100.0,
        "mom": 0.998,
        "rate_center": "zero",
    }
    if hmm != expected:
        raise RuntimeError(f"exp209 HMM contract changed: {hmm} != {expected}")
    return hmm


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    step: float,
    n_rates: int,
    rate_span: float,
    df: float,
    emission: str,
    sigma_mode: str,
    band_pad: float,
    rate_center: str,
    **_: Any,
) -> dict[str, Any]:
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    if not required_horizontal.issubset(horizontal.columns):
        raise ValueError(
            f"horizontal missing {sorted(required_horizontal - set(horizontal.columns))}"
        )
    if not required_typewell.issubset(typewell.columns):
        raise ValueError(
            f"typewell missing {sorted(required_typewell - set(typewell.columns))}"
        )
    if "TVT" in horizontal.columns:
        raise ValueError("prepare_hmm_inputs forbids unknown-suffix true TVT")

    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4:
        raise ValueError("known TVT_input prefix must contain at least four rows")
    if len(eval_rows) == 0:
        raise ValueError("well has no unknown suffix")

    cal_a, cal_b, robust_sigma, init_rate, rate_rows, valid_steps = prefix_stats(
        horizontal, typewell_tvt, typewell_gr, tail_n=30
    )
    if sigma_mode == "std":
        known_tvt = known["TVT_input"].to_numpy(np.float64)
        typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
        residual = known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
        gr_sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))
        cal_a_use, cal_b_use = 1.0, 0.0
    else:
        gr_sigma = robust_sigma
        cal_a_use, cal_b_use = cal_a, cal_b

    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - float(band_pad))
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + float(band_pad))
    grid = np.arange(grid_min, grid_max + float(step), float(step), dtype=np.float64)
    gr_grid = cal_a_use * np.interp(grid, typewell_tvt, typewell_gr) + cal_b_use

    md = eval_rows["MD"].to_numpy(np.float64)
    z = eval_rows["Z"].to_numpy(np.float64)
    raw_gr = eval_rows["GR"].to_numpy(np.float64)
    gr_fill = float(np.nanmean(typewell_gr))
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[eval_rows.index]
    )
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))
    zscore = (gr[:, None] - gr_grid[None, :]) / gr_sigma
    if emission == "t":
        emission_ll = (
            -0.5 * (float(df) + 1.0) * np.log1p(zscore**2 / float(df))
        ).astype(np.float32)
    elif emission == "gauss":
        emission_ll = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    else:
        raise ValueError(f"unsupported emission={emission}")
    if rate_center == "zero":
        span = max(float(rate_span), abs(init_rate) + 0.04)
        rates = np.linspace(-span, span, int(n_rates), dtype=np.float64)
    else:
        rates = init_rate + np.linspace(-float(rate_span), float(rate_span), int(n_rates))
    return {
        "emission_ll": emission_ll,
        "dm": dm,
        "dz": dz,
        "grid": grid,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / float(step)),
        "r0": float(init_rate),
        "eval_index": eval_rows.index.to_numpy(np.int64),
        "eval_id": (
            eval_rows["id"].astype(str).to_numpy()
            if "id" in eval_rows.columns
            else None
        ),
        "raw_gr_missing": ~np.isfinite(raw_gr),
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "last_known_z": float(last["Z"]),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir": init_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "cal_a": cal_a,
        "cal_b": cal_b,
    }


# %% [markdown]
# ## 5. Exact forward-backward with message sufficient statistics
#
# `cur` and `alpha` follow the exp270 parity implementation exactly.  Extra
# arrays are normalized readouts only; they are never fed back into the HMM.

# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_message_sufficient_statistics(
    em,
    dm,
    dz,
    sp,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    r0,
    r0_sig,
    lam,
    mom,
):
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)

    alpha = np.full((t_count, p_count, r_count), neg, np.float32)
    prev = np.full((p_count, r_count), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - r0) / r0_sig
            prev[p_i, r_i] = np.float32(lp0 - 0.5 * dr * dr)

    tmp = np.empty((p_count, r_count), np.float32)
    predictive = np.empty((p_count, r_count), np.float32)
    cur = np.empty((p_count, r_count), np.float32)

    predictive_pos_mass = np.zeros((t_count, p_count), np.float32)
    predictive_pos_r1 = np.zeros((t_count, p_count), np.float32)
    predictive_pos_r2 = np.zeros((t_count, p_count), np.float32)
    predictive_rate_mass = np.zeros((t_count, r_count), np.float32)
    filtered_pos_mass = np.zeros((t_count, p_count), np.float32)
    filtered_pos_r1 = np.zeros((t_count, p_count), np.float32)
    filtered_pos_r2 = np.zeros((t_count, p_count), np.float32)
    filtered_rate_mass = np.zeros((t_count, r_count), np.float32)
    predictive_logsum_minus_max = np.zeros(t_count, np.float64)
    filtered_logsum_minus_max = np.zeros(t_count, np.float64)
    filtered_logsum_score = np.zeros(t_count, np.float64)

    for t_i in range(t_count):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)

        for p_i in prange(p_count):
            for r2 in range(r_count):
                best = neg
                k0 = max(r2 - 1, 0)
                k1 = min(r2 + 1, r_count - 1)
                for r_i in range(k0, k1 + 1):
                    value = prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r_i in range(k0, k1 + 1):
                        total += np.exp(
                            prev[p_i, r_i]
                            + rate_log_kernel[r_i, r2 - r_i + 1]
                            - best
                        )
                    tmp[p_i, r2] = np.float32(best + np.log(total))
                else:
                    tmp[p_i, r2] = neg

        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(
                np.sum(np.exp(position_log_kernel - kernel_max))
            )
            position_log_kernel -= log_norm
            for p2 in range(p_count):
                best = neg
                for k_i in range(5):
                    p1 = p2 - (b0 - 2 + k_i)
                    if 0 <= p1 < p_count:
                        value = tmp[p1, r2] + position_log_kernel[k_i]
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p1 = p2 - (b0 - 2 + k_i)
                        if 0 <= p1 < p_count:
                            total += np.exp(
                                tmp[p1, r2] + position_log_kernel[k_i] - best
                            )
                    pre_emission_value = best + np.log(total)
                    predictive[p2, r2] = np.float32(pre_emission_value)
                    cur[p2, r2] = np.float32(
                        pre_emission_value + lam * em[t_i, p2]
                    )
                else:
                    predictive[p2, r2] = neg
                    cur[p2, r2] = neg

        predictive_best = neg
        filtered_best = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                if predictive[p_i, r_i] > predictive_best:
                    predictive_best = predictive[p_i, r_i]
                if cur[p_i, r_i] > filtered_best:
                    filtered_best = cur[p_i, r_i]
        predictive_total = 0.0
        filtered_total = 0.0
        for p_i in range(p_count):
            for r_i in range(r_count):
                predictive_total += np.exp(
                    predictive[p_i, r_i] - predictive_best
                )
                filtered_total += np.exp(cur[p_i, r_i] - filtered_best)
        predictive_logsum_minus_max[t_i] = np.log(predictive_total)
        filtered_logsum_minus_max[t_i] = np.log(filtered_total)
        filtered_logsum_score[t_i] = float(filtered_best) + np.log(filtered_total)

        for p_i in range(p_count):
            for r_i in range(r_count):
                predictive_probability = (
                    np.exp(predictive[p_i, r_i] - predictive_best)
                    / predictive_total
                )
                filtered_probability = (
                    np.exp(cur[p_i, r_i] - filtered_best) / filtered_total
                )
                rate = rates[r_i]
                predictive_pos_mass[t_i, p_i] += np.float32(
                    predictive_probability
                )
                predictive_pos_r1[t_i, p_i] += np.float32(
                    predictive_probability * rate
                )
                predictive_pos_r2[t_i, p_i] += np.float32(
                    predictive_probability * rate * rate
                )
                predictive_rate_mass[t_i, r_i] += np.float32(
                    predictive_probability
                )
                filtered_pos_mass[t_i, p_i] += np.float32(filtered_probability)
                filtered_pos_r1[t_i, p_i] += np.float32(
                    filtered_probability * rate
                )
                filtered_pos_r2[t_i, p_i] += np.float32(
                    filtered_probability * rate * rate
                )
                filtered_rate_mass[t_i, r_i] += np.float32(
                    filtered_probability
                )
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]

    best = np.float32(neg)
    for p_i in range(p_count):
        for r_i in range(r_count):
            if alpha[t_count - 1, p_i, r_i] > best:
                best = alpha[t_count - 1, p_i, r_i]
    total = 0.0
    for p_i in range(p_count):
        for r_i in range(r_count):
            total += np.exp(alpha[t_count - 1, p_i, r_i] - best)
    loglik = float(best) + np.log(total)

    post_p = np.zeros((t_count, p_count), np.float64)
    beta_next = np.zeros((p_count, r_count), np.float32)
    values = alpha[t_count - 1] + beta_next
    best = np.max(values)
    total = 0.0
    for p_i in range(p_count):
        acc = 0.0
        for r_i in range(r_count):
            acc += np.exp(values[p_i, r_i] - best)
        post_p[t_count - 1, p_i] = acc
        total += acc
    post_p[t_count - 1] /= total
    for p_i in range(p_count):
        for r_i in range(r_count):
            probability = np.exp(values[p_i, r_i] - best) / total
            alpha[t_count - 1, p_i, r_i] = np.float32(probability)

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)

        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(
                np.sum(np.exp(position_log_kernel - kernel_max))
            )
            position_log_kernel -= log_norm
            for p1 in range(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if 0 <= p2 < p_count:
                        value = (
                            position_log_kernel[k_i]
                            + lam * em[t_i, p2]
                            + beta_next[p2, r2]
                        )
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p2 = p1 + (b0 - 2 + k_i)
                        if 0 <= p2 < p_count:
                            total += np.exp(
                                position_log_kernel[k_i]
                                + lam * em[t_i, p2]
                                + beta_next[p2, r2]
                                - best
                            )
                    beta_tmp[p1, r2] = np.float32(best + np.log(total))
                else:
                    beta_tmp[p1, r2] = neg

        for p_i in prange(p_count):
            for r_i in range(r_count):
                best = neg
                k0 = max(r_i - 1, 0)
                k1 = min(r_i + 1, r_count - 1)
                for r2 in range(k0, k1 + 1):
                    value = (
                        rate_log_kernel[r_i, r2 - r_i + 1]
                        + beta_tmp[p_i, r2]
                    )
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1]
                            + beta_tmp[p_i, r2]
                            - best
                        )
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg

        values = alpha[t_i - 1] + beta_cur
        best = np.max(values)
        total = 0.0
        for p_i in range(p_count):
            acc = 0.0
            for r_i in range(r_count):
                acc += np.exp(values[p_i, r_i] - best)
            post_p[t_i - 1, p_i] = acc
            total += acc
        post_p[t_i - 1] /= total
        for p_i in range(p_count):
            for r_i in range(r_count):
                probability = np.exp(values[p_i, r_i] - best) / total
                alpha[t_i - 1, p_i, r_i] = np.float32(probability)
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]

    return (
        post_p,
        alpha,
        loglik,
        predictive_pos_mass,
        predictive_pos_r1,
        predictive_pos_r2,
        predictive_rate_mass,
        filtered_pos_mass,
        filtered_pos_r1,
        filtered_pos_r2,
        filtered_rate_mass,
        predictive_logsum_minus_max,
        filtered_logsum_minus_max,
        filtered_logsum_score,
    )


def hmm_common_arguments(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
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
        float(hmm["mom"]),
    )


def shrink_smoothed_joint(
    joint: np.ndarray,
    post_p: np.ndarray,
    rates: np.ndarray,
) -> dict[str, np.ndarray]:
    joint_normalization = np.sum(joint, axis=(1, 2), dtype=np.float64)
    if np.any(joint_normalization <= 0.0):
        raise RuntimeError("smoothed joint has non-positive row mass")
    smooth_pos_mass = np.asarray(post_p, dtype=np.float32)
    smooth_pos_r1 = np.einsum(
        "tpr,r->tp",
        joint,
        rates,
        dtype=np.float64,
        optimize=True,
    )
    smooth_pos_r2 = np.einsum(
        "tpr,r->tp",
        joint,
        rates**2,
        dtype=np.float64,
        optimize=True,
    )
    smooth_rate_mass = np.sum(joint, axis=1, dtype=np.float64)
    smooth_pos_r1 /= joint_normalization[:, None]
    smooth_pos_r2 /= joint_normalization[:, None]
    smooth_rate_mass /= joint_normalization[:, None]
    return {
        "pos_mass": smooth_pos_mass,
        "pos_r1": smooth_pos_r1.astype(np.float32),
        "pos_r2": smooth_pos_r2.astype(np.float32),
        "rate_mass": smooth_rate_mass.astype(np.float32),
    }


def normalize_forward_stage_statistics(
    pos_mass: np.ndarray,
    pos_r1: np.ndarray,
    pos_r2: np.ndarray,
    rate_mass: np.ndarray,
) -> None:
    position_normalization = np.sum(pos_mass, axis=1, dtype=np.float64)
    rate_normalization = np.sum(rate_mass, axis=1, dtype=np.float64)
    if np.any(position_normalization <= 0.0) or np.any(rate_normalization <= 0.0):
        raise RuntimeError("forward message sufficient statistics lost all mass")
    pos_mass /= position_normalization[:, None]
    pos_r1 /= position_normalization[:, None]
    pos_r2 /= position_normalization[:, None]
    rate_mass /= rate_normalization[:, None]


def run_current_hmm_messages(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    emission = np.asarray(prepared["emission_ll"], dtype=np.float32)
    result = _hmm2_message_sufficient_statistics(
        emission,
        *hmm_common_arguments(prepared, hmm),
    )
    (
        post_p,
        joint,
        log_likelihood,
        predictive_pos_mass,
        predictive_pos_r1,
        predictive_pos_r2,
        predictive_rate_mass,
        filtered_pos_mass,
        filtered_pos_r1,
        filtered_pos_r2,
        filtered_rate_mass,
        predictive_logsum_minus_max,
        filtered_logsum_minus_max,
        filtered_logsum_score,
    ) = result
    normalize_forward_stage_statistics(
        predictive_pos_mass,
        predictive_pos_r1,
        predictive_pos_r2,
        predictive_rate_mass,
    )
    normalize_forward_stage_statistics(
        filtered_pos_mass,
        filtered_pos_r1,
        filtered_pos_r2,
        filtered_rate_mass,
    )
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    smooth = shrink_smoothed_joint(joint, post_p, rates)
    del joint
    posterior_mean = np.asarray(post_p, dtype=np.float64) @ np.asarray(
        prepared["grid"], dtype=np.float64
    )
    stages = {
        "predictive": {
            "pos_mass": predictive_pos_mass,
            "pos_r1": predictive_pos_r1,
            "pos_r2": predictive_pos_r2,
            "rate_mass": predictive_rate_mass,
            "logsum_minus_max": predictive_logsum_minus_max,
        },
        "filtered": {
            "pos_mass": filtered_pos_mass,
            "pos_r1": filtered_pos_r1,
            "pos_r2": filtered_pos_r2,
            "rate_mass": filtered_rate_mass,
            "logsum_minus_max": filtered_logsum_minus_max,
            "logsum_score": filtered_logsum_score,
        },
        "smoothed": smooth,
    }
    normalization = {
        stage: {
            "position_max_abs_error": float(
                np.max(
                    np.abs(
                        np.sum(values["pos_mass"], axis=1, dtype=np.float64) - 1.0
                    )
                )
            ),
            "rate_max_abs_error": float(
                np.max(
                    np.abs(
                        np.sum(values["rate_mass"], axis=1, dtype=np.float64) - 1.0
                    )
                )
            ),
        }
        for stage, values in stages.items()
    }
    message_sha = array_bundle_sha256(
        posterior_mean=posterior_mean.astype(np.float32),
        predictive_pos_mass=predictive_pos_mass,
        predictive_pos_r1=predictive_pos_r1,
        predictive_pos_r2=predictive_pos_r2,
        predictive_rate_mass=predictive_rate_mass,
        filtered_pos_mass=filtered_pos_mass,
        filtered_pos_r1=filtered_pos_r1,
        filtered_pos_r2=filtered_pos_r2,
        filtered_rate_mass=filtered_rate_mass,
        smooth_pos_mass=smooth["pos_mass"],
        smooth_pos_r1=smooth["pos_r1"],
        smooth_pos_r2=smooth["pos_r2"],
        smooth_rate_mass=smooth["rate_mass"],
    )
    return {
        "posterior_mean": posterior_mean,
        "stages": stages,
        "log_likelihood": float(log_likelihood),
        "normalization": normalization,
        "message_sha256": message_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


# %% [markdown]
# ## 6. Target-free message freeze and transition moments

# %%
def validate_same_pass_parity(
    prepared: Mapping[str, Any],
    saved: pd.DataFrame,
    posterior_mean: np.ndarray,
    atol_ft: float,
) -> dict[str, Any]:
    ordered = saved.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    raw_rows = np.asarray(prepared["eval_index"], dtype=np.int64)
    if not np.array_equal(ordered["row_idx"].to_numpy(np.int64), raw_rows):
        raise RuntimeError("raw suffix row_idx differs from exp270")
    if prepared["eval_id"] is not None and not np.array_equal(
        ordered["id"].astype(str).to_numpy(),
        np.asarray(prepared["eval_id"]).astype(str),
    ):
        raise RuntimeError("raw suffix id differs from exp270")
    regenerated = np.asarray(posterior_mean, dtype=np.float64).astype(np.float32)
    reference = ordered["posterior_mean"].to_numpy(np.float64).astype(np.float32)
    difference = np.abs(regenerated.astype(np.float64) - reference.astype(np.float64))
    result = {
        "rows": len(difference),
        "max_abs_diff_ft": float(difference.max(initial=0.0)),
        "mean_abs_diff_ft": float(difference.mean()) if len(difference) else 0.0,
        "atol_ft": float(atol_ft),
    }
    result["passed"] = bool(result["max_abs_diff_ft"] <= float(atol_ft))
    if not result["passed"]:
        raise RuntimeError(f"same-pass exp270 parity failed: {result}")
    return result


def rate_transition_destination_mass(
    source_mass: np.ndarray,
    rates: np.ndarray,
    dm: float,
    sig_r: float,
    mom: float,
) -> tuple[np.ndarray, float]:
    source = np.asarray(source_mass, dtype=np.float64)
    source = source / source.sum()
    step = float(rates[1] - rates[0])
    rate_var_cells = (float(sig_r) * math.sqrt(float(dm)) / step) ** 2
    destination = np.zeros(len(rates), dtype=np.float64)
    for source_index, rate in enumerate(rates):
        mean_move = -(1.0 - float(mom)) * float(rate) * float(dm) / step
        p_plus = max(0.5 * (rate_var_cells + mean_move), 1.0e-12)
        p_minus = max(0.5 * (rate_var_cells - mean_move), 1.0e-12)
        total = p_plus + p_minus
        if total > 0.9:
            p_plus *= 0.9 / total
            p_minus *= 0.9 / total
        probabilities = (p_minus, 1.0 - p_plus - p_minus, p_plus)
        for offset, probability in zip((-1, 0, 1), probabilities, strict=True):
            destination_index = source_index + offset
            if 0 <= destination_index < len(rates):
                destination[destination_index] += (
                    source[source_index] * float(probability)
                )
    survival = float(destination.sum())
    if survival <= 0.0:
        raise RuntimeError("rate transition lost all probability mass")
    return destination / survival, survival


def current_position_kernel_moments(
    rates: np.ndarray,
    dm: float,
    dz: float,
    step: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.empty(len(rates), dtype=np.float64)
    variances = np.empty(len(rates), dtype=np.float64)
    intended = np.asarray(rates, dtype=np.float64) * float(dm) - float(dz)
    sigma_position = max(float(sig_p), 0.35 * float(step))
    offsets = np.arange(-2, 3, dtype=np.int64)
    for rate_index, mu in enumerate(intended):
        center = int(np.floor(mu / float(step) + 0.5))
        displacements = (center + offsets).astype(np.float64) * float(step)
        weights = np.exp(-0.5 * ((displacements - mu) / sigma_position) ** 2)
        weights /= weights.sum()
        mean = float(np.sum(weights * displacements))
        means[rate_index] = mean
        variances[rate_index] = float(
            np.sum(weights * (displacements - mean) ** 2)
        )
    return means, variances, intended


def transition_moment_ledger(
    prepared: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, np.ndarray]],
    hmm: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    dm = np.asarray(prepared["dm"], dtype=np.float64)
    dz = np.asarray(prepared["dz"], dtype=np.float64)
    filtered_rate_mass = np.asarray(stages["filtered"]["rate_mass"], dtype=np.float64)
    initial = np.exp(
        -0.5 * ((rates - float(prepared["r0"])) / float(hmm["r0_sig"])) ** 2
    )
    initial /= initial.sum()
    count = len(dm)
    fields = {
        "current_expected_displacement_ft": np.empty(count, dtype=np.float64),
        "current_expected_variance_ft2": np.empty(count, dtype=np.float64),
        "exact_mean_expected_displacement_ft": np.empty(count, dtype=np.float64),
        "current_minus_exact_mean_ft": np.empty(count, dtype=np.float64),
        "rate_transition_survival_mass": np.empty(count, dtype=np.float64),
        "destination_rate_mean": np.empty(count, dtype=np.float64),
        "destination_rate_variance": np.empty(count, dtype=np.float64),
    }
    for row in range(count):
        source = initial if row == 0 else filtered_rate_mass[row - 1]
        destination, survival = rate_transition_destination_mass(
            source,
            rates,
            float(dm[row]),
            float(hmm["sig_r"]),
            float(hmm["mom"]),
        )
        current_mean_by_rate, current_var_by_rate, intended = (
            current_position_kernel_moments(
                rates,
                float(dm[row]),
                float(dz[row]),
                float(hmm["step"]),
                float(hmm["sig_p"]),
            )
        )
        current_mean = float(np.sum(destination * current_mean_by_rate))
        exact_mean = float(np.sum(destination * intended))
        total_variance = float(
            np.sum(
                destination
                * (
                    current_var_by_rate
                    + (current_mean_by_rate - current_mean) ** 2
                )
            )
        )
        destination_rate_mean = float(np.sum(destination * rates))
        fields["current_expected_displacement_ft"][row] = current_mean
        fields["current_expected_variance_ft2"][row] = total_variance
        fields["exact_mean_expected_displacement_ft"][row] = exact_mean
        fields["current_minus_exact_mean_ft"][row] = current_mean - exact_mean
        fields["rate_transition_survival_mass"][row] = survival
        fields["destination_rate_mean"][row] = destination_rate_mean
        fields["destination_rate_variance"][row] = float(
            np.sum(destination * (rates - destination_rate_mean) ** 2)
        )
    return fields


def freeze_target_free_messages(
    *,
    well: str,
    prepared: Mapping[str, Any],
    saved: pd.DataFrame,
    messages: Mapping[str, Any],
    hmm: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> dict[str, Any]:
    posterior_mean = np.asarray(messages["posterior_mean"], dtype=np.float64)
    parity = validate_same_pass_parity(
        prepared,
        saved,
        posterior_mean,
        float(get_nested(config, "validation.parity_atol_ft")),
    )
    maximum_normalization_error = max(
        max(
            float(stage["position_max_abs_error"]),
            float(stage["rate_max_abs_error"]),
        )
        for stage in messages["normalization"].values()
    )
    normalization_atol = float(get_nested(config, "validation.normalization_atol"))
    if maximum_normalization_error > normalization_atol:
        raise RuntimeError(
            f"{well}: message normalization error "
            f"{maximum_normalization_error} > {normalization_atol}"
        )
    transition = transition_moment_ledger(prepared, messages["stages"], hmm)
    prediction_sha = array_bundle_sha256(
        posterior_mean=posterior_mean.astype(np.float32),
        row_idx=np.asarray(prepared["eval_index"], dtype=np.int64),
    )
    message_sha = str(messages["message_sha256"])
    transition_sha = array_bundle_sha256(
        **{key: np.asarray(value, dtype=np.float64) for key, value in transition.items()}
    )
    ledger.freeze(well, prediction_sha, message_sha)
    return {
        "parity": parity,
        "maximum_normalization_error": maximum_normalization_error,
        "prediction_sha256": prediction_sha,
        "message_sha256": message_sha,
        "transition_sha256": transition_sha,
        "transition": transition,
    }


# %% [markdown]
# ## 7. Truth-late basin and hidden-rate readout

# %%
def load_episode_rows_late(
    well: str,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    spec = get_nested(config, "data.persistent_episodes")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    # Read only after the current well's prediction and message SHA are frozen.
    frame = pd.read_csv(path)
    selected = (
        frame.loc[frame["well"].astype(str) == str(well)]
        .sort_values("start_row_idx", kind="mergesort")
        .reset_index(drop=True)
    )
    ledger.record_episode_late(well, len(selected))
    return selected


def load_truth_late(
    well: str,
    raw_dir: Path,
    prepared: Mapping[str, Any],
    ledger: LeakageLedger,
) -> np.ndarray:
    path = raw_dir / f"{well}__horizontal_well.csv"
    truth = pd.read_csv(path, usecols=["TVT"])
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    if row_idx.max(initial=-1) >= len(truth):
        raise RuntimeError(f"{well}: suffix row index exceeds truth rows")
    values = pd.to_numeric(truth.iloc[row_idx]["TVT"], errors="coerce").to_numpy(
        np.float64
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{well}: suffix truth contains non-finite values")
    ledger.record_truth_late(well, len(values))
    return values


def path_rate(
    path: np.ndarray,
    dm: np.ndarray,
    dz: np.ndarray,
    last_tvt: float,
) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    previous = np.concatenate([[float(last_tvt)], values[:-1]])
    return (values - previous + np.asarray(dz, dtype=np.float64)) / np.asarray(
        dm, dtype=np.float64
    )


def interval_sum_by_row(
    matrix: np.ndarray,
    grid: np.ndarray,
    centers: np.ndarray,
    radius: float,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    left = np.searchsorted(grid, np.asarray(centers) - float(radius), side="left")
    right = (
        np.searchsorted(grid, np.asarray(centers) + float(radius), side="right")
        - 1
    )
    valid = (left < len(grid)) & (right >= 0) & (left <= right)
    left = np.clip(left, 0, len(grid) - 1)
    right = np.clip(right, 0, len(grid) - 1)
    cumulative = np.cumsum(values, axis=1, dtype=np.float64)
    rows = np.arange(len(values), dtype=np.int64)
    result = cumulative[rows, right]
    has_left = left > 0
    result[has_left] -= cumulative[rows[has_left], left[has_left] - 1]
    result[~valid] = 0.0
    return result


def rate_neighborhood_mass(
    rate_mass: np.ndarray,
    rates: np.ndarray,
    centers: np.ndarray,
    cells: int,
) -> np.ndarray:
    values = np.asarray(rate_mass, dtype=np.float64)
    center_index = np.argmin(
        np.abs(rates[None, :] - np.asarray(centers)[:, None]),
        axis=1,
    )
    cumulative = np.cumsum(values, axis=1, dtype=np.float64)
    left = np.maximum(center_index - int(cells), 0)
    right = np.minimum(center_index + int(cells), len(rates) - 1)
    rows = np.arange(len(values), dtype=np.int64)
    result = cumulative[rows, right]
    has_left = left > 0
    result[has_left] -= cumulative[rows[has_left], left[has_left] - 1]
    return result


def safe_variance(mean: np.ndarray, second_moment: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.asarray(second_moment, dtype=np.float64)
        - np.asarray(mean, dtype=np.float64) ** 2,
        0.0,
    )


def safe_logit(values: np.ndarray, floor: float) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), floor, 1.0 - floor)
    return np.log(probability) - np.log1p(-probability)


def stage_readout(
    *,
    stage_name: str,
    stage: Mapping[str, np.ndarray],
    grid: np.ndarray,
    rates: np.ndarray,
    position_centers: Mapping[str, np.ndarray],
    rate_centers: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    pos_mass = np.asarray(stage["pos_mass"], dtype=np.float64)
    pos_r1 = np.asarray(stage["pos_r1"], dtype=np.float64)
    pos_r2 = np.asarray(stage["pos_r2"], dtype=np.float64)
    rate_mass = np.asarray(stage["rate_mass"], dtype=np.float64)
    position_mean = pos_mass @ grid
    position_second = pos_mass @ (grid**2)
    rate_mean = rate_mass @ rates
    rate_second = rate_mass @ (rates**2)
    covariance = pos_r1 @ grid - position_mean * rate_mean
    position_cells = int(get_nested(config, "audit.position_edge_cells"))
    rate_cells = int(get_nested(config, "audit.rate_edge_cells"))
    result: dict[str, np.ndarray] = {
        f"{stage_name}__position_mean": position_mean,
        f"{stage_name}__position_std": np.sqrt(
            safe_variance(position_mean, position_second)
        ),
        f"{stage_name}__rate_mean": rate_mean,
        f"{stage_name}__rate_std": np.sqrt(safe_variance(rate_mean, rate_second)),
        f"{stage_name}__position_rate_covariance": covariance,
        f"{stage_name}__position_edge_mass": (
            pos_mass[:, :position_cells].sum(axis=1)
            + pos_mass[:, -position_cells:].sum(axis=1)
        ),
        f"{stage_name}__rate_edge_mass": (
            rate_mass[:, :rate_cells].sum(axis=1)
            + rate_mass[:, -rate_cells:].sum(axis=1)
        ),
    }
    if "logsum_minus_max" in stage:
        result[f"{stage_name}__logsum_minus_max"] = np.asarray(
            stage["logsum_minus_max"], dtype=np.float64
        )
    if "logsum_score" in stage:
        result[f"{stage_name}__logsum_score"] = np.asarray(
            stage["logsum_score"], dtype=np.float64
        )

    radius = float(get_nested(config, "audit.basin_radius_ft"))
    rate_cells_near = int(get_nested(config, "audit.rate_neighborhood_cells"))
    floor = float(get_nested(config, "audit.probability_floor"))
    for label, centers in position_centers.items():
        basin_mass = interval_sum_by_row(pos_mass, grid, centers, radius)
        basin_r1 = interval_sum_by_row(pos_r1, grid, centers, radius)
        basin_r2 = interval_sum_by_row(pos_r2, grid, centers, radius)
        conditional_mean = np.divide(
            basin_r1,
            basin_mass,
            out=np.full(len(basin_mass), np.nan, dtype=np.float64),
            where=basin_mass > floor,
        )
        conditional_second = np.divide(
            basin_r2,
            basin_mass,
            out=np.full(len(basin_mass), np.nan, dtype=np.float64),
            where=basin_mass > floor,
        )
        conditional_variance = np.maximum(
            conditional_second - conditional_mean**2,
            0.0,
        )
        result[f"{stage_name}__{label}_position_mass"] = basin_mass
        result[f"{stage_name}__{label}_position_logit"] = safe_logit(
            basin_mass, floor
        )
        result[f"{stage_name}__{label}_conditional_rate_mean"] = conditional_mean
        result[f"{stage_name}__{label}_conditional_rate_std"] = np.sqrt(
            conditional_variance
        )
        result[f"{stage_name}__{label}_rate_near_mass"] = rate_neighborhood_mass(
            rate_mass,
            rates,
            rate_centers[label],
            rate_cells_near,
        )
    return result


def all_stage_readouts(
    *,
    messages: Mapping[str, Any],
    prepared: Mapping[str, Any],
    posterior_mean: np.ndarray,
    viterbi: np.ndarray,
    truth: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    dm = np.asarray(prepared["dm"], dtype=np.float64)
    dz = np.asarray(prepared["dz"], dtype=np.float64)
    last_tvt = float(prepared["last_known_tvt"])
    rate_centers = {
        "truth": path_rate(truth, dm, dz, last_tvt),
        "mean": path_rate(posterior_mean, dm, dz, last_tvt),
        "viterbi": path_rate(viterbi, dm, dz, last_tvt),
    }
    position_centers = {
        "truth": np.asarray(truth, dtype=np.float64),
        "mean": np.asarray(posterior_mean, dtype=np.float64),
        "viterbi": np.asarray(viterbi, dtype=np.float64),
    }
    readout: dict[str, np.ndarray] = {}
    for stage_name, stage in messages["stages"].items():
        readout.update(
            stage_readout(
                stage_name=stage_name,
                stage=stage,
                grid=np.asarray(prepared["grid"], dtype=np.float64),
                rates=np.asarray(prepared["rates"], dtype=np.float64),
                position_centers=position_centers,
                rate_centers=rate_centers,
                config=config,
            )
        )

    for label in ("truth", "mean", "viterbi"):
        readout[f"emission__{label}_position_logit_delta"] = (
            readout[f"filtered__{label}_position_logit"]
            - readout[f"predictive__{label}_position_logit"]
        )
        readout[f"beta__{label}_position_logit_delta"] = (
            readout[f"smoothed__{label}_position_logit"]
            - readout[f"filtered__{label}_position_logit"]
        )
    for left, right, key in (
        ("truth", "mean", "truth_vs_mean"),
        ("truth", "viterbi", "truth_vs_viterbi"),
        ("viterbi", "mean", "viterbi_vs_mean"),
    ):
        for stage_name in ("predictive", "filtered", "smoothed"):
            readout[f"{stage_name}__{key}_position_logit"] = (
                readout[f"{stage_name}__{left}_position_logit"]
                - readout[f"{stage_name}__{right}_position_logit"]
            )
        readout[f"emission__{key}_logit_delta"] = (
            readout[f"filtered__{key}_position_logit"]
            - readout[f"predictive__{key}_position_logit"]
        )
        readout[f"beta__{key}_logit_delta"] = (
            readout[f"smoothed__{key}_position_logit"]
            - readout[f"filtered__{key}_position_logit"]
        )
    return readout, rate_centers


def direct_emission_readout(
    prepared: Mapping[str, Any],
    position_centers: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    emission = np.asarray(prepared["emission_ll"], dtype=np.float64)
    rows = np.arange(len(emission), dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for label, centers in position_centers.items():
        index = np.rint((np.asarray(centers) - grid[0]) / (grid[1] - grid[0])).astype(
            np.int64
        )
        inside = (index >= 0) & (index < len(grid))
        clipped = np.clip(index, 0, len(grid) - 1)
        values = emission[rows, clipped]
        values[~inside] = np.nan
        result[f"emission_ll__{label}"] = values
        result[f"{label}_position_support_inside"] = inside
    result["emission_ll__truth_minus_mean"] = (
        result["emission_ll__truth"] - result["emission_ll__mean"]
    )
    result["emission_ll__truth_minus_viterbi"] = (
        result["emission_ll__truth"] - result["emission_ll__viterbi"]
    )
    return result


def compensating_rate_readout(
    prepared: Mapping[str, Any],
    truth_rate: np.ndarray,
    filtered_rate_mean: np.ndarray,
    hmm: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    dm = np.asarray(prepared["dm"], dtype=np.float64)
    dz = np.asarray(prepared["dz"], dtype=np.float64)
    true_displacement = np.asarray(truth_rate) * dm - dz
    best_rate = np.empty(len(dm), dtype=np.float64)
    best_distance = np.empty(len(dm), dtype=np.float64)
    best_edge = np.empty(len(dm), dtype=bool)
    for row in range(len(dm)):
        means, _, _ = current_position_kernel_moments(
            rates,
            float(dm[row]),
            float(dz[row]),
            float(hmm["step"]),
            float(hmm["sig_p"]),
        )
        index = int(np.argmin(np.abs(means - true_displacement[row])))
        best_rate[row] = rates[index]
        best_distance[row] = abs(rates[index] - float(filtered_rate_mean[row]))
        best_edge[row] = index in (0, len(rates) - 1)
    return {
        "compensating_rate_state": best_rate,
        "compensating_rate_distance_from_filtered_mean": best_distance,
        "compensating_rate_is_edge": best_edge,
    }


# %% [markdown]
# ## 8. Episode attribution

# %%
def build_well_readout(
    *,
    well: str,
    saved: pd.DataFrame,
    prepared: Mapping[str, Any],
    messages: Mapping[str, Any],
    frozen: Mapping[str, Any],
    truth: np.ndarray,
    config: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> pd.DataFrame:
    ordered = saved.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    posterior_mean = np.asarray(messages["posterior_mean"], dtype=np.float64)
    viterbi = ordered["topk_path_1"].to_numpy(np.float64)
    marginal_map = ordered["marginal_map"].to_numpy(np.float64)
    readout, rate_centers = all_stage_readouts(
        messages=messages,
        prepared=prepared,
        posterior_mean=posterior_mean,
        viterbi=viterbi,
        truth=truth,
        config=config,
    )
    position_centers = {
        "truth": np.asarray(truth, dtype=np.float64),
        "mean": posterior_mean,
        "viterbi": viterbi,
    }
    emission = direct_emission_readout(prepared, position_centers)
    compensating = compensating_rate_readout(
        prepared,
        rate_centers["truth"],
        readout["filtered__rate_mean"],
        hmm,
    )
    row_idx = np.asarray(prepared["eval_index"], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "well": str(well),
            "id": ordered["id"].astype(str).to_numpy(),
            "row_idx": row_idx,
            "suffix_offset": np.arange(len(row_idx), dtype=np.int64),
            "fold": ordered["fold"].to_numpy(np.int8),
            "md_since": ordered["md_since"].to_numpy(np.float64),
            "tvt_true": np.asarray(truth, dtype=np.float64),
            "posterior_mean": posterior_mean,
            "marginal_map": marginal_map,
            "global_viterbi": viterbi,
            "mean_error_ft": posterior_mean - np.asarray(truth, dtype=np.float64),
            "viterbi_error_ft": viterbi - np.asarray(truth, dtype=np.float64),
            "raw_gr_missing": np.asarray(prepared["raw_gr_missing"], dtype=bool),
            "true_rate": rate_centers["truth"],
            "mean_path_rate": rate_centers["mean"],
            "viterbi_path_rate": rate_centers["viterbi"],
            "truth_rate_support_inside": (
                (rate_centers["truth"] >= float(np.min(prepared["rates"])))
                & (rate_centers["truth"] <= float(np.max(prepared["rates"])))
            ),
            "message_sha256": str(frozen["message_sha256"]),
            "prediction_sha256": str(frozen["prediction_sha256"]),
        }
    )
    for collection in (
        readout,
        frozen["transition"],
        emission,
        compensating,
    ):
        for key, values in collection.items():
            frame[key] = np.asarray(values)
    frame["current_kernel_quantization_bias_ft"] = frame[
        "current_minus_exact_mean_ft"
    ]
    frame["true_displacement_ft"] = (
        frame["true_rate"].to_numpy(np.float64)
        * np.asarray(prepared["dm"], dtype=np.float64)
        - np.asarray(prepared["dz"], dtype=np.float64)
    )
    return frame


def fraction(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)) if len(array) else math.nan


def first_true_offset(mask: np.ndarray, start_offset: int = 0) -> float:
    index = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(index) == 0:
        return math.nan
    return float(int(index[0]) + int(start_offset))


def classify_episode(
    episode_rows: pd.DataFrame,
    *,
    thresholds: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    effect = float(thresholds["log_odds_effect"])
    dominant = float(thresholds["dominant_row_fraction"])
    position_outside = ~episode_rows["truth_position_support_inside"].to_numpy(bool)
    rate_outside = ~episode_rows["truth_rate_support_inside"].to_numpy(bool)
    support_fraction = fraction(position_outside | rate_outside)

    filtered_truth_vs_mean = episode_rows[
        "filtered__truth_vs_mean_position_logit"
    ].to_numpy(np.float64)
    smoothed_truth_vs_mean = episode_rows[
        "smoothed__truth_vs_mean_position_logit"
    ].to_numpy(np.float64)
    beta_delta = episode_rows["beta__truth_vs_mean_logit_delta"].to_numpy(
        np.float64
    )
    emission_delta = episode_rows[
        "emission__truth_vs_mean_logit_delta"
    ].to_numpy(np.float64)
    predictive_truth_vs_mean = episode_rows[
        "predictive__truth_vs_mean_position_logit"
    ].to_numpy(np.float64)
    missing = episode_rows["raw_gr_missing"].to_numpy(bool)

    reversal_fraction = fraction(
        (filtered_truth_vs_mean > 0.0)
        & (smoothed_truth_vs_mean < 0.0)
        & (beta_delta < -effect)
    )
    observed_mask = ~missing
    raw_alias_fraction = (
        fraction(emission_delta[observed_mask] < -effect)
        if observed_mask.any()
        else 0.0
    )
    imputed_alias_fraction = (
        fraction(emission_delta[missing] < -effect) if missing.any() else 0.0
    )
    forward_fraction = fraction(predictive_truth_vs_mean < -effect)
    mean_rmse = float(
        np.sqrt(np.mean(episode_rows["mean_error_ft"].to_numpy(np.float64) ** 2))
    )
    viterbi_rmse = float(
        np.sqrt(
            np.mean(episode_rows["viterbi_error_ft"].to_numpy(np.float64) ** 2)
        )
    )
    viterbi_gain = mean_rmse - viterbi_rmse
    mean_viterbi_gap = float(
        np.mean(
            np.abs(
                episode_rows["posterior_mean"].to_numpy(np.float64)
                - episode_rows["global_viterbi"].to_numpy(np.float64)
            )
        )
    )
    multiplicity_gap = float(
        np.median(
            episode_rows["filtered__logsum_minus_max"].to_numpy(np.float64)
        )
    )

    if support_fraction >= float(thresholds["state_support_fraction"]):
        cause = "state_support_shortage"
    elif reversal_fraction >= dominant:
        cause = "backward_smoothing_reversal"
    elif raw_alias_fraction >= dominant:
        cause = "raw_gr_alias"
    elif imputed_alias_fraction >= dominant and fraction(missing) > 0.0:
        cause = "imputation_alias"
    elif forward_fraction >= dominant:
        cause = "forward_transition_prior_hysteresis"
    elif (
        viterbi_gain >= float(thresholds["viterbi_rmse_gain_ft"])
        and mean_viterbi_gap >= float(thresholds["mean_viterbi_gap_ft"])
        and multiplicity_gap >= float(thresholds["logsum_max_gap"])
    ):
        cause = "sum_product_path_multiplicity"
    else:
        cause = "mixed_or_unresolved"
    diagnostics = {
        "state_support_shortage_fraction": support_fraction,
        "backward_reversal_fraction": reversal_fraction,
        "raw_gr_alias_fraction_observed": raw_alias_fraction,
        "imputation_alias_fraction_missing": imputed_alias_fraction,
        "forward_hysteresis_fraction": forward_fraction,
        "raw_gr_missing_fraction": fraction(missing),
        "mean_rmse_ft": mean_rmse,
        "viterbi_rmse_ft": viterbi_rmse,
        "viterbi_rmse_gain_ft": viterbi_gain,
        "mean_viterbi_abs_gap_ft": mean_viterbi_gap,
        "filtered_logsum_minus_max_median": multiplicity_gap,
    }
    return cause, diagnostics


def summarize_episode(
    *,
    episode: pd.Series,
    well_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = int(episode["start_row_idx"])
    end = int(episode["end_row_idx_exclusive"])
    rows = well_frame.loc[
        (well_frame["row_idx"] >= start) & (well_frame["row_idx"] < end)
    ].copy()
    expected_rows = int(episode["rows"])
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{episode['episode_id']}: rows={len(rows)}/{expected_rows}"
        )
    rows.insert(0, "episode_id", str(episode["episode_id"]))
    rows.insert(4, "episode_row_offset", np.arange(len(rows), dtype=np.int64))

    thresholds = get_nested(config, "audit.cause_thresholds")
    cause, diagnostics = classify_episode(rows, thresholds=thresholds)
    truth_mass = well_frame["smoothed__truth_position_mass"].to_numpy(np.float64)
    start_suffix = int(rows["suffix_offset"].iloc[0])
    escape_threshold = 0.10
    recapture_threshold = 0.50
    after_start = truth_mass[start_suffix:]
    escape_relative = first_true_offset(after_start < escape_threshold)
    if math.isfinite(escape_relative):
        escape_suffix = start_suffix + int(escape_relative)
        recapture_relative = first_true_offset(
            truth_mass[escape_suffix:] >= recapture_threshold
        )
        recapture_suffix = (
            float(escape_suffix + int(recapture_relative))
            if math.isfinite(recapture_relative)
            else math.nan
        )
    else:
        escape_suffix = math.nan
        recapture_suffix = math.nan

    sse = float(np.sum(rows["mean_error_ft"].to_numpy(np.float64) ** 2))
    summary: dict[str, Any] = {
        "episode_id": str(episode["episode_id"]),
        "well": str(episode["well"]),
        "fold": int(rows["fold"].iloc[0]),
        "start_row_idx": start,
        "end_row_idx_exclusive": end,
        "rows": len(rows),
        "episode_sse": sse,
        "cause": cause,
        "truth_basin_first_escape_suffix_offset": escape_suffix,
        "truth_basin_first_recapture_suffix_offset": recapture_suffix,
        "truth_basin_escape_after_episode_start_rows": (
            escape_suffix - start_suffix if math.isfinite(escape_suffix) else math.nan
        ),
        "truth_basin_recapture_after_escape_rows": (
            recapture_suffix - escape_suffix
            if math.isfinite(recapture_suffix) and math.isfinite(escape_suffix)
            else math.nan
        ),
        "predictive_truth_mass_mean": float(
            rows["predictive__truth_position_mass"].mean()
        ),
        "filtered_truth_mass_mean": float(
            rows["filtered__truth_position_mass"].mean()
        ),
        "smoothed_truth_mass_mean": float(
            rows["smoothed__truth_position_mass"].mean()
        ),
        "predictive_truth_rate_mass_mean": float(
            rows["predictive__truth_rate_near_mass"].mean()
        ),
        "filtered_truth_rate_mass_mean": float(
            rows["filtered__truth_rate_near_mass"].mean()
        ),
        "smoothed_truth_rate_mass_mean": float(
            rows["smoothed__truth_rate_near_mass"].mean()
        ),
        "emission_truth_vs_mean_logit_delta_mean": float(
            rows["emission__truth_vs_mean_logit_delta"].mean()
        ),
        "beta_truth_vs_mean_logit_delta_mean": float(
            rows["beta__truth_vs_mean_logit_delta"].mean()
        ),
        "current_minus_exact_mean_ft_mean": float(
            rows["current_minus_exact_mean_ft"].mean()
        ),
        "position_rate_covariance_mean": float(
            rows["smoothed__position_rate_covariance"].mean()
        ),
        **diagnostics,
    }
    for key in (
        "emission_evidence_class",
        "observed_emission_evidence_class",
        "affine_observed_emission_evidence_class",
        "viterbi_recovery_class",
    ):
        if key in episode.index:
            summary[f"prior_audit__{key}"] = episode[key]
    return rows, summary


def cause_summary_frame(episodes: pd.DataFrame) -> pd.DataFrame:
    total_sse = float(episodes["episode_sse"].sum())
    grouped = (
        episodes.groupby("cause", sort=True)
        .agg(
            episodes=("episode_id", "size"),
            wells=("well", "nunique"),
            rows=("rows", "sum"),
            episode_sse=("episode_sse", "sum"),
            mean_rmse_ft=("mean_rmse_ft", "mean"),
            viterbi_rmse_gain_ft=("viterbi_rmse_gain_ft", "mean"),
            predictive_truth_mass_mean=("predictive_truth_mass_mean", "mean"),
            filtered_truth_mass_mean=("filtered_truth_mass_mean", "mean"),
            smoothed_truth_mass_mean=("smoothed_truth_mass_mean", "mean"),
        )
        .reset_index()
    )
    grouped["episode_fraction"] = grouped["episodes"] / len(episodes)
    grouped["sse_fraction"] = grouped["episode_sse"] / total_sse
    return grouped.sort_values(
        ["episode_sse", "cause"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


# %% [markdown]
# ## 9. Kaggle CPU orchestration

# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP408_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "exp408 full HMM audit must run on Kaggle CPU; local execution is disabled"
    )


def open_deterministic_gzip_text(path: Path) -> tuple[Any, Any, Any]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=raw,
        compresslevel=1,
        mtime=0,
    )
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def combined_manifest_sha(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    selected = [
        {field: row[field] for field in fields}
        for row in sorted(rows, key=lambda item: str(item["well"]))
    ]
    return hashlib.sha256(stable_json_bytes({"rows": selected})).hexdigest()


def run_full_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    counts = validate_execution_contract(config)
    set_num_threads(int(get_nested(config, "execution.numba_num_threads")))
    ledger = LeakageLedger()
    hmm = fixed_hmm_kwargs(config)
    target_wells, target_manifest = load_target_wells(config, ledger)
    target_set = set(target_wells)
    mode_bank, mode_manifest = load_target_free_mode_bank(
        config, target_set, ledger
    )
    folds, fold_manifest = load_target_free_folds(config, target_set, ledger)
    mode_bank = attach_target_free_folds(mode_bank, folds)
    del folds
    gc.collect()
    raw_dir = train_data_dir(config)
    missing_raw = [
        well
        for well in target_wells
        if not (raw_dir / f"{well}__horizontal_well.csv").is_file()
        or not (raw_dir / f"{well}__typewell.csv").is_file()
    ]
    if missing_raw:
        raise RuntimeError(f"missing raw wells: {missing_raw[:10]}")

    output = output_dir()
    prefix = EXPERIMENT_NAME
    row_path = output / f"{prefix}_row_ledger.csv.gz"
    raw_stream, compressed_stream, text_stream = open_deterministic_gzip_text(
        row_path
    )
    wrote_header = False
    episode_summaries: list[dict[str, Any]] = []
    well_manifests: list[dict[str, Any]] = []
    total_episode_rows = 0
    groups = mode_bank.groupby("well", sort=False).indices
    hard_runtime = float(get_nested(config, "execution.hard_runtime_seconds"))
    hard_peak_rss = float(get_nested(config, "execution.hard_peak_rss_gb"))
    progress_every = int(get_nested(config, "execution.progress_every_wells"))

    try:
        for well_index, well in enumerate(target_wells, start=1):
            well_started = time.perf_counter()
            saved = mode_bank.iloc[groups[well]].copy().reset_index(drop=True)
            horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
            prepared = prepare_hmm_inputs(horizontal, typewell, **hmm)
            messages = run_current_hmm_messages(prepared, hmm)
            frozen = freeze_target_free_messages(
                well=well,
                prepared=prepared,
                saved=saved,
                messages=messages,
                hmm=hmm,
                config=config,
                ledger=ledger,
            )

            # Truth and episode boundaries are intentionally unavailable above.
            episodes = load_episode_rows_late(well, config, ledger)
            truth = load_truth_late(well, raw_dir, prepared, ledger)
            well_frame = build_well_readout(
                well=well,
                saved=saved,
                prepared=prepared,
                messages=messages,
                frozen=frozen,
                truth=truth,
                config=config,
                hmm=hmm,
            )
            well_episode_rows = 0
            for _, episode in episodes.iterrows():
                row_frame, episode_summary = summarize_episode(
                    episode=episode,
                    well_frame=well_frame,
                    config=config,
                )
                row_frame.to_csv(
                    text_stream,
                    index=False,
                    header=not wrote_header,
                )
                wrote_header = True
                well_episode_rows += len(row_frame)
                total_episode_rows += len(row_frame)
                episode_summaries.append(episode_summary)

            well_manifest = {
                "well": well,
                "suffix_rows": len(saved),
                "episodes": len(episodes),
                "episode_rows": well_episode_rows,
                "prefix_rows": int(prepared["prefix_rows"]),
                "grid_points": len(prepared["grid"]),
                "rate_states": len(prepared["rates"]),
                "rate_min": float(np.min(prepared["rates"])),
                "rate_max": float(np.max(prepared["rates"])),
                "prefix_initial_rate": float(prepared["prefix_ir"]),
                "prefix_sigma_gr": float(prepared["prefix_sigma"]),
                "log_likelihood": float(messages["log_likelihood"]),
                "parity_max_abs_diff_ft": float(
                    frozen["parity"]["max_abs_diff_ft"]
                ),
                "normalization_max_abs_error": float(
                    frozen["maximum_normalization_error"]
                ),
                "prediction_sha256": str(frozen["prediction_sha256"]),
                "message_sha256": str(frozen["message_sha256"]),
                "transition_sha256": str(frozen["transition_sha256"]),
                "hmm_elapsed_seconds": float(messages["elapsed_seconds"]),
                "well_elapsed_seconds": float(time.perf_counter() - well_started),
                "peak_rss_gb": peak_rss_gb(),
            }
            well_manifests.append(well_manifest)
            elapsed = float(time.perf_counter() - started)
            if elapsed > hard_runtime:
                raise RuntimeError(f"runtime hard guard exceeded: {elapsed}")
            if peak_rss_gb() > hard_peak_rss:
                raise MemoryError(f"peak RSS hard guard exceeded: {peak_rss_gb()}")
            if well_index % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "exp408_progress",
                            "well_index": well_index,
                            "target_wells": len(target_wells),
                            "well": well,
                            "suffix_rows": len(saved),
                            "episode_rows": well_episode_rows,
                            "hmm_seconds": messages["elapsed_seconds"],
                            "elapsed_seconds": elapsed,
                            "peak_rss_gb": peak_rss_gb(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            del (
                saved,
                horizontal,
                typewell,
                prepared,
                messages,
                frozen,
                episodes,
                truth,
                well_frame,
            )
            gc.collect()
    finally:
        text_stream.flush()
        text_stream.close()
        compressed_stream.close()
        raw_stream.close()

    episode_frame = pd.DataFrame(episode_summaries).sort_values(
        ["well", "start_row_idx"], kind="mergesort"
    )
    well_frame = pd.DataFrame(well_manifests).sort_values("well", kind="mergesort")
    expected_episodes = int(get_nested(config, "validation.expected_episodes"))
    expected_episode_rows = int(
        get_nested(config, "validation.expected_episode_rows")
    )
    if len(episode_frame) != expected_episodes:
        raise RuntimeError(f"episodes={len(episode_frame)}/{expected_episodes}")
    if total_episode_rows != expected_episode_rows:
        raise RuntimeError(
            f"episode rows={total_episode_rows}/{expected_episode_rows}"
        )
    if int(well_frame["suffix_rows"].sum()) != int(
        get_nested(config, "validation.expected_suffix_rows")
    ):
        raise RuntimeError("processed suffix row count differs from contract")
    if not wrote_header:
        raise RuntimeError("row ledger is empty")
    cause_frame = cause_summary_frame(episode_frame)

    episode_artifact = write_csv(
        output / f"{prefix}_episode_summary.csv", episode_frame
    )
    cause_artifact = write_csv(output / f"{prefix}_cause_summary.csv", cause_frame)
    well_artifact = write_csv(output / f"{prefix}_well_manifest.csv", well_frame)
    row_artifact = {
        "path": str(row_path),
        "raw_sha256": sha256_file(row_path),
        "decompressed_sha256": sha256_decompressed_csv(row_path),
        "rows": total_episode_rows,
    }
    prediction_sha = combined_manifest_sha(
        well_manifests, ("well", "prediction_sha256")
    )
    message_sha = combined_manifest_sha(
        well_manifests, ("well", "message_sha256", "transition_sha256")
    )
    input_manifest = {
        "target_wells": target_manifest,
        "exp270_mode_bank": mode_manifest,
        "exp226_fold_identity": fold_manifest,
        "raw_train_dir": str(raw_dir),
        "counts": {
            "target_wells": len(target_wells),
            "suffix_rows": int(well_frame["suffix_rows"].sum()),
            "episodes": len(episode_frame),
            "episode_rows": total_episode_rows,
        },
        "leakage": {
            "target_well_scope_reads": ledger.target_well_scope_reads,
            "target_free_rows": ledger.target_free_rows,
            "truth_rows_before_well_freeze": ledger.truth_rows_before_well_freeze,
            "episode_rows_before_well_freeze": ledger.episode_rows_before_well_freeze,
            "truth_rows_after_well_freeze": ledger.truth_rows_after_well_freeze,
            "episode_rows_after_well_freeze": ledger.episode_rows_after_well_freeze,
            "frozen_wells": len(ledger.frozen_wells),
        },
    }
    input_artifact = write_json(
        output / f"{prefix}_input_manifest.json", input_manifest
    )
    parity_max = float(well_frame["parity_max_abs_diff_ft"].max())
    normalization_max = float(well_frame["normalization_max_abs_error"].max())
    technical = {
        "well_count": len(well_frame) == len(target_wells),
        "suffix_rows": int(well_frame["suffix_rows"].sum())
        == int(get_nested(config, "validation.expected_suffix_rows")),
        "episode_count": len(episode_frame) == expected_episodes,
        "episode_rows": total_episode_rows == expected_episode_rows,
        "prediction_parity": parity_max
        <= float(get_nested(config, "validation.parity_atol_ft")),
        "message_normalization": normalization_max
        <= float(get_nested(config, "validation.normalization_atol")),
        "truth_reads_before_well_freeze": ledger.truth_rows_before_well_freeze == 0,
        "episode_reads_before_well_freeze": (
            ledger.episode_rows_before_well_freeze == 0
        ),
        "finite_episode_metrics": bool(
            np.isfinite(
                episode_frame[
                    [
                        "episode_sse",
                        "mean_rmse_ft",
                        "predictive_truth_mass_mean",
                        "filtered_truth_mass_mean",
                        "smoothed_truth_mass_mean",
                    ]
                ].to_numpy(np.float64)
            ).all()
        ),
        "runtime": float(time.perf_counter() - started) <= hard_runtime,
        "peak_rss": peak_rss_gb() <= hard_peak_rss,
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "complete" if all(technical.values()) else "technical_fail",
        "execution_contract": counts,
        "technical_gates": technical,
        "counts": input_manifest["counts"],
        "parity_max_abs_diff_ft": parity_max,
        "normalization_max_abs_error": normalization_max,
        "prediction_manifest_sha256": prediction_sha,
        "message_manifest_sha256": message_sha,
        "cause_summary": cause_frame.to_dict(orient="records"),
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_rss_gb": peak_rss_gb(),
            "versions": runtime_versions(),
            "cpu_only": True,
            "numba_num_threads": int(
                get_nested(config, "execution.numba_num_threads")
            ),
        },
        "artifacts": {
            "row_ledger": row_artifact,
            "episode_summary": episode_artifact,
            "cause_summary": cause_artifact,
            "well_manifest": well_artifact,
            "input_manifest": input_artifact,
        },
    }
    summary_artifact = write_json(output / f"{prefix}_summary.json", summary)
    summary["artifacts"]["summary"] = summary_artifact
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": summary["status"],
        "validation": {
            "strategy": get_nested(config, "validation.strategy"),
            "cv": None,
            "lb": None,
        },
        "execution_contract": counts,
        "technical_gates": technical,
        "result": {
            "counts": summary["counts"],
            "parity_max_abs_diff_ft": parity_max,
            "normalization_max_abs_error": normalization_max,
            "prediction_manifest_sha256": prediction_sha,
            "message_manifest_sha256": message_sha,
            "cause_summary": summary["cause_summary"],
            "runtime": summary["runtime"],
        },
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 10. Metrics and generated artifacts
#
# The final cell prints the fixed execution count before starting and writes a
# row ledger, episode summary, cause summary, well manifest, input manifest, and
# summary JSON.  No model, inference file, or submission is produced.

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    EXECUTION_COUNTS = validate_execution_contract(CONFIG)
    print(
        json.dumps(
            {
                "event": "exp408_start",
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "run_stage": get_nested(CONFIG, "execution.run_stage"),
                "execution_counts": EXECUTION_COUNTS,
                "runtime": "kaggle_cpu",
                "inference": False,
                "submission": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    SUMMARY = run_full_audit(CONFIG)

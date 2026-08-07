# %% [markdown]
# # exp270 exact HMM posterior mode candidate audit
#
# This CPU-only train-side notebook keeps the raw exp209 exact-HMM scientific
# contract fixed and changes only posterior readout. The canonical `train`
# notebook aggregates two target-free well shards; `train_variant0/1` generate
# those disjoint shards. Unknown-suffix true TVT is attached only after every
# candidate path for a well is frozen.

# %% [markdown]
# ## Contents
# 1. Imports and experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Raw-well and known-prefix HMM inputs
# 4. Exact exp209 forward-backward kernel
# 5. Exact joint-state top-K decoder
# 6. Target-free posterior-mode generation
# 7. Candidate and oracle audit helpers
# 8. Shard generation and aggregate orchestration
# 9. Setup and input preflight
# 10. Generate a shard or aggregate the audit
# 11. Metrics and artifact summary

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
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import numba
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - Kaggle image includes numba.
    numba = None
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any):
        del args, kwargs

        def decorator(function):
            return function

        return decorator

    def prange(*args: int):
        return range(*args)

    def set_num_threads(_: int) -> None:
        return None


EXPERIMENT_NAME = "exp270_exact_hmm_posterior_mode_candidate_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TOP_K = 5
BLOCK_ROWS = (128, 256, 512)
PATH_COLUMNS = tuple(f"topk_path_{rank}" for rank in range(1, TOP_K + 1))
FIXED_CANDIDATES = ("posterior_mean", "marginal_map")
ALL_CANDIDATES = FIXED_CANDIDATES + PATH_COLUMNS
RUN_KIND_OVERRIDE = "shard1"
EXECUTE_NOTEBOOK = os.environ.get("EXP270_IMPORT_ONLY", "0") != "1"


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp270 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def train_data_dir(config: dict[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed_candidates = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed_candidates:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        first_horizontal = next(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"), None)
        if first_horizontal is not None:
            return first_horizontal.parent
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
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


def memory_usage_mb() -> dict[str, float | None]:
    current_rss_mb: float | None = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_rss_mb = float(line.split()[1]) / 1024.0
                break
    except (FileNotFoundError, OSError, ValueError):
        pass
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mb = peak / 1024.0 if platform.system() != "Darwin" else peak / (1024.0**2)
    return {"rss_mb": current_rss_mb, "peak_rss_mb": peak_rss_mb}


def log_stage(stage: str, started: float, **details: Any) -> None:
    payload = {
        "event": "exp270_stage",
        "stage": str(stage),
        "elapsed_seconds": float(time.time() - started),
        **memory_usage_mb(),
        **details,
    }
    print(json.dumps(to_jsonable(payload), sort_keys=True), flush=True)


def update_array_bundle_header(
    digest: Any,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> None:
    digest.update(name.encode())
    digest.update(str(np.dtype(dtype)).encode())
    digest.update(np.asarray(shape, dtype=np.int64).tobytes())


def array_bundle_sha256_from_binary_parts(
    *,
    candidate_path: Path,
    row_idx_path: Path,
    rows: int,
    candidate_count: int,
    chunk_bytes: int,
) -> str:
    specifications = {
        "candidates": (candidate_path, np.dtype(np.float32), (rows, candidate_count)),
        "row_idx": (row_idx_path, np.dtype(np.int64), (rows,)),
    }
    digest = hashlib.sha256()
    for name in sorted(specifications):
        path, dtype, shape = specifications[name]
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"{name} binary size mismatch: {path.stat().st_size}/{expected_bytes}"
            )
        update_array_bundle_header(digest, name, dtype, shape)
        with path.open("rb") as file_pointer:
            for chunk in iter(lambda: file_pointer.read(int(chunk_bytes)), b""):
                digest.update(chunk)
    return digest.hexdigest()


def array_bundle_sha256_from_frame(
    frame: pd.DataFrame,
    *,
    chunk_rows: int,
) -> str:
    specifications = {
        "candidates": (
            list(ALL_CANDIDATES),
            np.dtype(np.float32),
            (len(frame), len(ALL_CANDIDATES)),
        ),
        "row_idx": (["row_idx"], np.dtype(np.int64), (len(frame),)),
    }
    digest = hashlib.sha256()
    for name in sorted(specifications):
        columns, dtype, shape = specifications[name]
        update_array_bundle_header(digest, name, dtype, shape)
        for start in range(0, len(frame), int(chunk_rows)):
            block = frame.iloc[start : start + int(chunk_rows)][columns]
            if name == "row_idx":
                array = block.iloc[:, 0].to_numpy(dtype=np.int64)
            else:
                array = block.to_numpy(dtype=np.float32)
            digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def write_dataframe_gzip_deterministic(
    path: Path,
    frame: pd.DataFrame,
    *,
    chunk_rows: int,
    compresslevel: int,
    mtime: int,
) -> None:
    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            compresslevel=int(compresslevel),
            mtime=int(mtime),
        ) as gzip_file:
            with io.TextIOWrapper(gzip_file, encoding="utf-8", newline="") as text_file:
                if frame.empty:
                    frame.to_csv(text_file, index=False)
                else:
                    for start in range(0, len(frame), int(chunk_rows)):
                        frame.iloc[start : start + int(chunk_rows)].to_csv(
                            text_file,
                            index=False,
                            header=start == 0,
                        )


def iter_frame_batches(
    frame: pd.DataFrame,
    columns: list[str],
    chunk_rows: int,
) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), int(chunk_rows)):
        yield frame.iloc[start : start + int(chunk_rows)][columns].reset_index(drop=True)


def iter_candidate_csv_batches(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        usecols=["id", "well", "posterior_mean"],
        dtype={"id": str, "well": str},
        chunksize=int(chunk_rows),
    )


def iter_control_batches(
    path: Path,
    control_column: str,
    selected_wells: set[str] | None,
    chunk_rows: int,
) -> Iterator[pd.DataFrame]:
    for frame in pd.read_csv(
        path,
        usecols=["id", "well", control_column],
        dtype={"id": str, "well": str},
        chunksize=int(chunk_rows),
    ):
        if selected_wells is not None:
            frame = frame.loc[frame["well"].isin(selected_wells)]
        if not frame.empty:
            yield frame.reset_index(drop=True)


def next_nonempty_batch(iterator: Iterator[pd.DataFrame]) -> pd.DataFrame | None:
    for frame in iterator:
        if not frame.empty:
            return frame.reset_index(drop=True)
    return None


def compare_ordered_posterior_batches(
    candidate_batches: Iterator[pd.DataFrame],
    control_batches: Iterator[pd.DataFrame],
    *,
    control_column: str,
    expected_rows: int,
    atol_ft: float,
) -> dict[str, Any]:
    candidate = next_nonempty_batch(candidate_batches)
    control = next_nonempty_batch(control_batches)
    candidate_offset = 0
    control_offset = 0
    compared = 0
    difference_sum = 0.0
    difference_max = 0.0
    while candidate is not None and control is not None:
        count = min(len(candidate) - candidate_offset, len(control) - control_offset)
        candidate_block = candidate.iloc[candidate_offset : candidate_offset + count]
        control_block = control.iloc[control_offset : control_offset + count]
        candidate_ids = candidate_block["id"].astype(str).to_numpy()
        control_ids = control_block["id"].astype(str).to_numpy()
        candidate_wells = candidate_block["well"].astype(str).to_numpy()
        control_wells = control_block["well"].astype(str).to_numpy()
        matching = (candidate_ids == control_ids) & (candidate_wells == control_wells)
        if not bool(matching.all()):
            local_index = int(np.flatnonzero(~matching)[0])
            raise ValueError(
                "exp209 ordered parity id mismatch at row "
                f"{compared + local_index}: candidate={candidate_ids[local_index]} "
                f"control={control_ids[local_index]}"
            )
        candidate_values = pd.to_numeric(
            candidate_block["posterior_mean"], errors="coerce"
        ).to_numpy(np.float64)
        control_values = pd.to_numeric(
            control_block[control_column], errors="coerce"
        ).to_numpy(np.float64)
        if not np.isfinite(candidate_values).all() or not np.isfinite(control_values).all():
            raise ValueError("exp209 ordered parity contains non-finite values")
        difference = np.abs(candidate_values - control_values)
        difference_sum += float(difference.sum(dtype=np.float64))
        difference_max = max(difference_max, float(difference.max(initial=0.0)))
        compared += count
        candidate_offset += count
        control_offset += count
        if candidate_offset == len(candidate):
            candidate = next_nonempty_batch(candidate_batches)
            candidate_offset = 0
        if control_offset == len(control):
            control = next_nonempty_batch(control_batches)
            control_offset = 0
    if candidate is not None or control is not None:
        raise ValueError("exp209 ordered parity row count differs between candidate and control")
    if compared != int(expected_rows):
        raise ValueError(f"exp209 ordered parity rows={compared}/{expected_rows}")
    parity = {
        "rows": compared,
        "max_abs_diff_ft": difference_max,
        "mean_abs_diff_ft": difference_sum / compared if compared else np.nan,
        "atol_ft": float(atol_ft),
        "passed": bool(difference_max <= float(atol_ft)),
        "alignment": "linear_ordered_id_well_chunks",
    }
    if not parity["passed"]:
        raise ValueError(f"exp209 posterior-mean parity failed: {parity}")
    return parity


def validate_posterior_mean_parity_batches(
    candidate_batches: Iterator[pd.DataFrame],
    config: dict[str, Any],
    *,
    expected_rows: int,
    selected_wells: set[str] | None,
) -> tuple[dict[str, Any], Path, str]:
    control_spec = get_nested(config, "data.exp209_hmm_control") or {}
    control_path = resolve_existing(
        str(control_spec["filename"]), [str(value) for value in control_spec["candidates"]]
    )
    control_decompressed_sha = require_decompressed_sha(
        control_path, str(control_spec.get("expected_decompressed_sha256") or "")
    )
    control_column = str(control_spec.get("prediction_column") or "hmm_mean_tvt")
    chunk_rows = int(get_nested(config, "execution.parity_chunksize_rows") or 100000)
    parity = compare_ordered_posterior_batches(
        candidate_batches,
        iter_control_batches(control_path, control_column, selected_wells, chunk_rows),
        control_column=control_column,
        expected_rows=expected_rows,
        atol_ft=float(control_spec.get("parity_atol_ft", 0.0)),
    )
    return parity, control_path, control_decompressed_sha


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_well_shard(well: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    key = f"exp270::well_shard::{well}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % shard_count


def resolve_existing(filename: str, candidates: list[str]) -> Path:
    checked: list[str] = []
    root = project_root()
    for raw in candidates:
        candidate = Path(raw)
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.stat().st_size > 0:
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename)):
            checked.append(str(path))
            if path.exists() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(f"Could not resolve {filename}; checked={checked}")


def require_decompressed_sha(path: Path, expected: str | None) -> str:
    actual = sha256_gzip_decompressed(path)
    if expected and actual != expected:
        raise ValueError(
            f"decompressed SHA mismatch for {path}: expected={expected} actual={actual}"
        )
    return actual


# %% [markdown]
# ## 3. Raw-well and known-prefix HMM inputs


# %%
def list_well_ids(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    wells: list[str] = []
    for path in sorted(root.glob("*__horizontal_well.csv")):
        well = path.stem.replace("__horizontal_well", "")
        if (root / f"{well}__typewell.csv").exists():
            wells.append(well)
    return wells


def load_well(well: str, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    horizontal = pd.read_csv(root / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(root / f"{well}__typewell.csv").sort_values("TVT").reset_index(drop=True)
    return horizontal, typewell


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


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


def fixed_hmm_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    hmm = get_nested(config, "model.hmm") or {}
    keys = (
        "step",
        "n_rates",
        "rate_span",
        "sig_r",
        "sig_p",
        "df",
        "emission",
        "lam",
        "sigma_mode",
        "start_sig",
        "r0_sig",
        "band_pad",
        "mom",
        "rate_center",
    )
    missing = [key for key in keys if key not in hmm]
    if missing:
        raise ValueError(f"model.hmm is missing fixed keys: {missing}")
    return {key: hmm[key] for key in keys}


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
    """Prepare target-free arrays using the unchanged exp209 preprocessing."""
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    if not required_horizontal.issubset(horizontal.columns):
        missing = sorted(required_horizontal - set(horizontal.columns))
        raise ValueError(f"horizontal missing {missing}")
    if not required_typewell.issubset(typewell.columns):
        missing = sorted(required_typewell - set(typewell.columns))
        raise ValueError(f"typewell missing {missing}")
    if "TVT" in horizontal.columns:
        raise ValueError("prepare_hmm_inputs forbids unknown-suffix true TVT")

    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4:
        raise ValueError("known TVT_input prefix must contain at least four rows")
    if len(eval_rows) == 0:
        raise ValueError("well has no unknown-suffix rows")

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
        emission_ll = (-0.5 * (float(df) + 1.0) * np.log1p(zscore**2 / float(df))).astype(
            np.float32
        )
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
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir": init_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "cal_a": cal_a,
        "cal_b": cal_b,
    }


# %% [markdown]
# ## 4. Exact exp209 forward-backward kernel


# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb(
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
    """Amerhu/exp209 exact forward-backward over (TVT position, dip-rate)."""
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
    cur = np.empty((p_count, r_count), np.float32)

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
                        total += np.exp(prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1] - best)
                    tmp[p_i, r2] = np.float32(best + np.log(total))
                else:
                    tmp[p_i, r2] = neg

        sigma_position = max(sig_p, 0.35 * sp)
        # Parallelize the independent rate slices in one region.  The previous
        # inner-p_count prange opened r_count parallel regions per time step,
        # which dominated runtime on the four-core Kaggle CPU worker.
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
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
                            total += np.exp(tmp[p1, r2] + position_log_kernel[k_i] - best)
                    cur[p2, r2] = np.float32(best + np.log(total) + lam * em[t_i, p2])
                else:
                    cur[p2, r2] = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
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
    post_p = np.zeros((t_count, p_count))
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
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p1 in range(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if 0 <= p2 < p_count:
                        value = position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2]
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
                    value = rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2] - best
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
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]
    return post_p, loglik


# %% [markdown]
# ## 5. Exact joint-state top-K decoder
#
# Backpointers are one byte per `(time, position, rate, rank)`. The byte stores
# one of 5 position offsets, one of 3 predecessor-rate offsets, and one of 5
# predecessor ranks (5 * 3 * 5 = 75 codes). This avoids materializing int32
# position/rate backpointer tensors.


# %%
@njit(cache=True, nogil=True)
def _insert_ranked(score_row, code_row, value, code):
    """Insert into a descending fixed-size list; equal scores keep loop order."""
    rank_count = len(score_row)
    for insert_at in range(rank_count):
        if value > score_row[insert_at]:
            for shift in range(rank_count - 1, insert_at, -1):
                score_row[shift] = score_row[shift - 1]
                code_row[shift] = code_row[shift - 1]
            score_row[insert_at] = value
            code_row[insert_at] = code
            return


@njit(cache=True, nogil=True, parallel=True)
def _hmm2_topk(
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
    top_k,
):
    """Exact global top-K joint paths under the exp209 HMM score."""
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)
    invalid = np.uint8(255)
    prev = np.full((p_count, r_count, top_k), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - r0) / r0_sig
            prev[p_i, r_i, 0] = np.float32(lp0 - 0.5 * dr * dr)

    cur = np.full((p_count, r_count, top_k), neg, np.float32)
    cur_code = np.full((p_count, r_count, top_k), invalid, np.uint8)
    tmp = np.full((p_count, r_count, top_k), neg, np.float32)
    tmp_code = np.full((p_count, r_count, top_k), invalid, np.uint8)
    merge_pointer = np.zeros((p_count, 5), np.int8)
    position_merge_pointer = np.zeros((r_count, p_count, 5), np.int8)
    if t_count > 1:
        backpointer = np.full((t_count - 1, p_count, r_count, top_k), invalid, np.uint8)
    else:
        backpointer = np.empty((0, p_count, r_count, top_k), np.uint8)

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

        # First merge the three predecessor-rate lists for fixed position/r2.
        for p_i in prange(p_count):
            for r2 in range(r_count):
                for source in range(3):
                    merge_pointer[p_i, source] = 0
                k0 = max(r2 - 1, 0)
                k1 = min(r2 + 1, r_count - 1)
                for output_rank in range(top_k):
                    best = neg
                    best_source = -1
                    best_previous_rank = -1
                    for r1 in range(k0, k1 + 1):
                        source = r1 - r2 + 1
                        previous_rank = int(merge_pointer[p_i, source])
                        if previous_rank >= top_k:
                            continue
                        value = prev[p_i, r1, previous_rank] + rate_log_kernel[r1, r2 - r1 + 1]
                        if value > best:
                            best = value
                            best_source = source
                            best_previous_rank = previous_rank
                    tmp[p_i, r2, output_rank] = best
                    if best_source >= 0:
                        tmp_code[p_i, r2, output_rank] = np.uint8(
                            best_source * top_k + best_previous_rank
                        )
                        merge_pointer[p_i, best_source] += 1
                    else:
                        tmp_code[p_i, r2, output_rank] = invalid

        # Then merge the five predecessor-position lists for fixed current state.
        sigma_position = max(sig_p, 0.35 * sp)
        # One parallel region over rate slices avoids launching a p_count
        # parfor 41 times per time step.  Position merge pointers include the
        # rate dimension so each parallel slice is independent.
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = position_log_kernel[0]
            for k_i in range(1, 5):
                if position_log_kernel[k_i] > kernel_max:
                    kernel_max = position_log_kernel[k_i]
            kernel_sum = 0.0
            for k_i in range(5):
                kernel_sum += np.exp(position_log_kernel[k_i] - kernel_max)
            log_norm = kernel_max + np.log(kernel_sum)
            for k_i in range(5):
                position_log_kernel[k_i] -= log_norm
            for p2 in range(p_count):
                for source in range(5):
                    position_merge_pointer[r2, p2, source] = 0
                for output_rank in range(top_k):
                    best = neg
                    best_position_code = -1
                    best_temporary_rank = -1
                    for position_code in range(5):
                        p1 = p2 - (b0 - 2 + position_code)
                        if p1 < 0 or p1 >= p_count:
                            continue
                        temporary_rank = int(position_merge_pointer[r2, p2, position_code])
                        if temporary_rank >= top_k:
                            continue
                        rate_rank_code = tmp_code[p1, r2, temporary_rank]
                        if rate_rank_code == invalid:
                            continue
                        value = (
                            tmp[p1, r2, temporary_rank]
                            + position_log_kernel[position_code]
                            + lam * em[t_i, p2]
                        )
                        if value > best:
                            best = value
                            best_position_code = position_code
                            best_temporary_rank = temporary_rank
                    cur[p2, r2, output_rank] = best
                    if best_position_code >= 0:
                        p1 = p2 - (b0 - 2 + best_position_code)
                        rate_rank_code = tmp_code[p1, r2, best_temporary_rank]
                        rate_code = int(rate_rank_code) // top_k
                        previous_rank = int(rate_rank_code) % top_k
                        cur_code[p2, r2, output_rank] = np.uint8(
                            ((best_position_code * 3 + rate_code) * top_k) + previous_rank
                        )
                        position_merge_pointer[r2, p2, best_position_code] += 1
                    else:
                        cur_code[p2, r2, output_rank] = invalid
                    if t_i > 0:
                        backpointer[t_i - 1, p2, r2, output_rank] = cur_code[p2, r2, output_rank]
        for p_i in range(p_count):
            for r_i in range(r_count):
                for rank in range(top_k):
                    prev[p_i, r_i, rank] = cur[p_i, r_i, rank]
                    cur[p_i, r_i, rank] = neg
                    cur_code[p_i, r_i, rank] = invalid

    terminal_scores = np.full(top_k, neg, np.float32)
    terminal_codes = np.full(top_k, -1, np.int64)
    for p_i in range(p_count):
        for r_i in range(r_count):
            for rank in range(top_k):
                value = prev[p_i, r_i, rank]
                flat_code = np.int64((p_i * r_count + r_i) * top_k + rank)
                for insert_at in range(top_k):
                    if value > terminal_scores[insert_at]:
                        for shift in range(top_k - 1, insert_at, -1):
                            terminal_scores[shift] = terminal_scores[shift - 1]
                            terminal_codes[shift] = terminal_codes[shift - 1]
                        terminal_scores[insert_at] = value
                        terminal_codes[insert_at] = flat_code
                        break

    position_paths = np.full((top_k, t_count), -1, np.int32)
    rate_paths = np.full((top_k, t_count), -1, np.int16)
    for output_rank in range(top_k):
        flat_code = terminal_codes[output_rank]
        if flat_code < 0:
            continue
        previous_rank = int(flat_code % top_k)
        state = int(flat_code // top_k)
        r2 = state % r_count
        p2 = state // r_count
        position_paths[output_rank, t_count - 1] = p2
        rate_paths[output_rank, t_count - 1] = r2
        for t_i in range(t_count - 1, 0, -1):
            code = backpointer[t_i - 1, p2, r2, previous_rank]
            if code == invalid:
                break
            previous_rank = int(code) % top_k
            packed = int(code) // top_k
            rate_code = packed % 3
            position_code = packed // 3
            r1 = r2 + rate_code - 1
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            p1 = p2 - (b0 - 2 + position_code)
            position_paths[output_rank, t_i - 1] = p1
            rate_paths[output_rank, t_i - 1] = r1
            p2, r2 = p1, r1
    return terminal_scores, position_paths, rate_paths


def estimate_backpointer_bytes(t_count: int, p_count: int, r_count: int, top_k: int) -> int:
    return max(int(t_count) - 1, 0) * int(p_count) * int(r_count) * int(top_k)


def deduplicate_tvt_paths(
    scores: np.ndarray,
    position_paths: np.ndarray,
    rate_paths: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate only by exact TVT position-index sequence, never by truth."""
    unique: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for joint_rank in range(len(scores)):
        position = np.ascontiguousarray(position_paths[joint_rank], dtype=np.int32)
        rate = np.ascontiguousarray(rate_paths[joint_rank], dtype=np.int16)
        if (position < 0).any() or (rate < 0).any():
            audit.append(
                {
                    "joint_rank": joint_rank + 1,
                    "status": "invalid",
                    "unique_rank": None,
                }
            )
            continue
        digest = hashlib.sha256(position.tobytes()).hexdigest()
        duplicate_of = seen.get(digest)
        if duplicate_of is None:
            unique_rank = len(unique) + 1
            seen[digest] = unique_rank
            unique.append(
                {
                    "unique_rank": unique_rank,
                    "joint_rank": joint_rank + 1,
                    "score": float(scores[joint_rank]),
                    "position": position,
                    "rate": rate,
                    "tvt_path_sha256": digest,
                }
            )
            status = "unique"
        else:
            unique_rank = duplicate_of
            status = "duplicate_tvt_path"
        audit.append(
            {
                "joint_rank": joint_rank + 1,
                "status": status,
                "unique_rank": unique_rank,
                "duplicate_of_unique_rank": duplicate_of,
                "score": float(scores[joint_rank]),
                "tvt_path_sha256": digest,
                "rate_path_sha256": hashlib.sha256(rate.tobytes()).hexdigest(),
            }
        )
    return unique, audit


# %% [markdown]
# ## 6. Target-free posterior-mode generation


# %%
def validate_scientific_contract(config: dict[str, Any]) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp270 route must be pf_beam")
    if get_nested(config, "lineage.parent") != (
        "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    ):
        raise ValueError("exp270 scientific parent must be exp209")
    if int(get_nested(config, "model.decoder.joint_top_k") or 0) != TOP_K:
        raise ValueError(f"joint_top_k must be exactly {TOP_K}")
    if get_nested(config, "model.decoder.deduplicate_by") != "tvt_grid_index_sequence":
        raise ValueError("top-K paths must be deduplicated by TVT grid-index sequence")
    if bool(get_nested(config, "model.decoder.backfill_after_dedup")):
        raise ValueError("approximate path backfill is forbidden")
    if bool(get_nested(config, "model.decoder.persist_rate_paths")) or bool(
        get_nested(config, "model.decoder.persist_full_posterior")
    ):
        raise ValueError("rate paths and full posterior persistence are forbidden")
    blocks = tuple(int(value) for value in get_nested(config, "audit.oracle_block_rows") or [])
    if blocks != BLOCK_ROWS:
        raise ValueError(f"oracle block rows must be exactly {BLOCK_ROWS}")
    forbidden_true = (
        bool(get_nested(config, "audit.persist_oracle_predictions")),
        bool(get_nested(config, "audit.persist_selector")),
        bool(get_nested(config, "audit.persist_candidate_blend")),
        bool(get_nested(config, "execution.gpu")),
        bool(get_nested(config, "execution.inference")),
        bool(get_nested(config, "execution.submission")),
    )
    if any(forbidden_true):
        raise ValueError(
            "oracle persistence, selector/blend, GPU, inference, and submission are forbidden"
        )
    if int(get_nested(config, "execution.active_hmm_variants") or 0) != 1:
        raise ValueError("exp270 must run exactly one raw HMM variant")
    if int(get_nested(config, "execution.lightgbm_config_count") or 0) != 0:
        raise ValueError("LightGBM is outside exp270")
    if int(get_nested(config, "execution.fold_count") or 0) != 0:
        raise ValueError("exp270 has no model folds")
    if int(get_nested(config, "execution.total_boosters") or 0) != 0:
        raise ValueError("exp270 has zero boosters")
    if int(get_nested(config, "execution.outer_workers") or 0) != 1:
        raise ValueError("exp270 processes one well at a time with outer_workers=1")
    if int(get_nested(config, "execution.shard_count") or 0) != 2:
        raise ValueError("exp270 recovery execution must use exactly two well shards")
    if int(get_nested(config, "execution.total_hmm_well_runs") or 0) != 773:
        raise ValueError("exp270 recovery execution must keep total HMM well-runs at 773")
    if not bool(get_nested(config, "execution.streaming_candidate_write")):
        raise ValueError("exp270 shard recovery requires streaming candidate writes")
    for key in (
        "execution.stream_flush_every_wells",
        "execution.parity_chunksize_rows",
        "execution.frame_write_chunksize_rows",
        "execution.binary_hash_chunk_bytes",
    ):
        if int(get_nested(config, key) or 0) <= 0:
            raise ValueError(f"exp270 requires positive {key}")
    compresslevel_value = get_nested(config, "execution.gzip_compresslevel")
    if compresslevel_value is None or not 0 <= int(compresslevel_value) <= 9:
        raise ValueError("exp270 gzip_compresslevel must be between 0 and 9")
    gzip_mtime = get_nested(config, "execution.gzip_mtime")
    if gzip_mtime is None or int(gzip_mtime) != 0:
        raise ValueError("exp270 deterministic gzip_mtime must be zero")
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise ValueError("exp270 shard recovery Kaggle push must be user approved")
    shard_specs = get_nested(config, "data.shard_outputs") or []
    if len(shard_specs) != 2:
        raise ValueError("exp270 requires exactly two configured shard outputs")
    if sum(int(spec.get("expected_rows", 0)) for spec in shard_specs) != int(
        get_nested(config, "validation.expected_rows") or 0
    ):
        raise ValueError("configured shard rows must sum to validation.expected_rows")
    if sum(int(spec.get("expected_wells", 0)) for spec in shard_specs) != int(
        get_nested(config, "validation.expected_wells") or 0
    ):
        raise ValueError("configured shard wells must sum to validation.expected_wells")
    if bool(get_nested(config, "execution.control_or_parent_retraining")):
        raise ValueError("parent/control model retraining is forbidden")


def run_hmm_posterior_modes(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate every mode candidate without accepting a true-TVT argument."""
    hmm = fixed_hmm_kwargs(config)
    prepared = prepare_hmm_inputs(horizontal_without_truth, typewell, **hmm)
    em = np.asarray(prepared["emission_ll"], dtype=np.float32)
    dm = np.asarray(prepared["dm"], dtype=np.float64)
    dz = np.asarray(prepared["dz"], dtype=np.float64)
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    rates = np.asarray(prepared["rates"], dtype=np.float64)
    top_k = int(get_nested(config, "model.decoder.joint_top_k"))
    backpointer_bytes = estimate_backpointer_bytes(em.shape[0], em.shape[1], len(rates), top_k)
    max_bytes = int(get_nested(config, "model.decoder.max_backpointer_bytes") or 0)
    if max_bytes and backpointer_bytes > max_bytes:
        raise MemoryError(
            f"top-K backpointer estimate {backpointer_bytes:,} exceeds guard {max_bytes:,}"
        )

    common = (
        em,
        dm,
        dz,
        float(hmm["step"]),
        rates,
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["r0"]),
        float(hmm["r0_sig"]),
        float(hmm["lam"]),
        float(hmm["mom"]),
    )
    posterior, log_likelihood = _hmm2_fb(*common)
    posterior_mean = posterior @ grid
    posterior_variance = posterior @ (grid**2) - posterior_mean**2
    posterior_std = np.sqrt(np.maximum(posterior_variance, 0.0))
    marginal_map_index = np.argmax(posterior, axis=1).astype(np.int32)
    marginal_map = grid[marginal_map_index]
    row_index = np.arange(len(posterior), dtype=np.int64)
    mode_mass = posterior[row_index, marginal_map_index]
    if posterior.shape[1] > 1:
        second_mass = np.partition(posterior, -2, axis=1)[:, -2]
    else:
        second_mass = np.zeros(len(posterior), dtype=np.float64)
    mode_gap = mode_mass - second_mass
    del posterior
    gc.collect()

    scores, position_paths, rate_paths = _hmm2_topk(*common, top_k)
    unique_paths, joint_audit = deduplicate_tvt_paths(
        np.asarray(scores), np.asarray(position_paths), np.asarray(rate_paths)
    )
    candidates = {
        "posterior_mean": np.asarray(posterior_mean, dtype=np.float64),
        "marginal_map": np.asarray(marginal_map, dtype=np.float64),
    }
    for rank in range(1, TOP_K + 1):
        column = f"topk_path_{rank}"
        candidates[column] = np.full(len(posterior_mean), np.nan, dtype=np.float64)
    for path in unique_paths:
        candidates[f"topk_path_{int(path['unique_rank'])}"] = grid[path["position"]]
    return {
        **prepared,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "marginal_map_index": marginal_map_index,
        "marginal_mode_mass": mode_mass,
        "marginal_mode_gap": mode_gap,
        "log_likelihood": float(log_likelihood),
        "candidates": candidates,
        "unique_paths": unique_paths,
        "joint_path_audit": joint_audit,
        "joint_scores": np.asarray(scores, dtype=np.float64),
        "backpointer_bytes": int(backpointer_bytes),
    }


def path_shape_diagnostics(
    position: np.ndarray,
    rate: np.ndarray,
    p_count: int,
    edge_margin: int,
) -> dict[str, Any]:
    position = np.asarray(position, dtype=np.int32)
    rate = np.asarray(rate, dtype=np.int16)
    step = np.diff(position.astype(np.float64))
    curvature = np.diff(position.astype(np.float64), n=2)
    return {
        "grid_edge_rate": float(
            np.mean((position <= edge_margin) | (position >= p_count - 1 - edge_margin))
        ),
        "rate_switch_rate": float(np.mean(np.diff(rate) != 0)) if len(rate) > 1 else 0.0,
        "position_step_abs_mean_grid": float(np.mean(np.abs(step))) if len(step) else 0.0,
        "position_step_abs_p99_grid": float(np.quantile(np.abs(step), 0.99)) if len(step) else 0.0,
        "curvature_abs_mean_grid": float(np.mean(np.abs(curvature))) if len(curvature) else 0.0,
        "curvature_abs_p99_grid": (
            float(np.quantile(np.abs(curvature), 0.99)) if len(curvature) else 0.0
        ),
    }


def candidate_storage_values(candidate: str, values: np.ndarray) -> np.ndarray:
    """Match the saved exp209 dtype only for its posterior-mean control."""
    dtype = np.float32 if candidate == "posterior_mean" else np.float64
    return np.asarray(values, dtype=dtype)


def build_candidate_rows_for_well(
    well: str,
    data_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Freeze target-free paths, then attach true TVT for diagnostics."""
    horizontal_path = data_dir / f"{well}__horizontal_well.csv"
    typewell_path = data_dir / f"{well}__typewell.csv"
    horizontal, typewell = load_well(well, data_dir)
    if "TVT" not in horizontal.columns:
        raise ValueError(f"{horizontal_path} is missing train-side TVT")
    generation_horizontal = horizontal.drop(columns=["TVT"]).copy()
    started = time.time()
    result = run_hmm_posterior_modes(generation_horizontal, typewell, config)
    eval_index = np.asarray(result["eval_index"], dtype=np.int64)

    # Candidate generation is complete above. Truth attachment starts here.
    true_tvt = pd.to_numeric(horizontal.loc[eval_index, "TVT"], errors="coerce").to_numpy(
        np.float64
    )
    if not np.isfinite(true_tvt).all():
        raise ValueError(f"non-finite evaluation target for well={well}")
    md_since = pd.to_numeric(horizontal.loc[eval_index, "MD"], errors="coerce").to_numpy(
        np.float64
    ) - float(result["last_known_md"])
    payload: dict[str, Any] = {
        "id": [f"{well}_{int(row)}" for row in eval_index],
        "well": str(well),
        "row_idx": eval_index,
        "true_tvt_readout_only": true_tvt.astype(np.float64),
        "last_known_tvt": np.float64(result["last_known_tvt"]),
        "md_since": md_since.astype(np.float32),
        "prefix_rows": np.int32(result["prefix_rows"]),
        "posterior_std": np.asarray(result["posterior_std"], dtype=np.float32),
        "marginal_mode_mass": np.asarray(result["marginal_mode_mass"], dtype=np.float32),
        "marginal_mode_gap": np.asarray(result["marginal_mode_gap"], dtype=np.float32),
    }
    for candidate in ALL_CANDIDATES:
        # exp209 casts every saved numeric column to float32. Normalize its
        # posterior-mean control to that persisted contract before strict
        # parity, while retaining float64 for the new mode-path readouts.
        payload[candidate] = candidate_storage_values(
            candidate, result["candidates"][candidate]
        )
    frame = pd.DataFrame(payload)
    if frame["id"].duplicated().any():
        raise ValueError(f"duplicate generated id for well={well}")

    edge_margin = int(get_nested(config, "audit.path_edge_margin_grid_points") or 0)
    path_rows: list[dict[str, Any]] = []
    unique_by_joint = {int(item["joint_rank"]): item for item in result["unique_paths"]}
    top_score = float(result["joint_scores"][0])
    for item in result["joint_path_audit"]:
        joint_rank = int(item["joint_rank"])
        unique_item = unique_by_joint.get(joint_rank)
        row = {
            "well": str(well),
            **item,
            "log_likelihood": float(result["log_likelihood"]),
            "path_log_posterior": float(item.get("score", np.nan))
            - float(result["log_likelihood"]),
            "score_gap_vs_top1": top_score - float(item.get("score", np.nan)),
            "unique_path_count_in_joint_top5": int(len(result["unique_paths"])),
            "backpointer_bytes": int(result["backpointer_bytes"]),
        }
        if unique_item is not None:
            row.update(
                path_shape_diagnostics(
                    unique_item["position"],
                    unique_item["rate"],
                    len(result["grid"]),
                    edge_margin,
                )
            )
        path_rows.append(row)

    pair_rows: list[dict[str, Any]] = []
    available = [
        candidate
        for candidate in ALL_CANDIDATES
        if np.isfinite(frame[candidate].to_numpy(np.float64)).all()
    ]
    for left, right in combinations(available, 2):
        difference = frame[left].to_numpy(np.float64) - frame[right].to_numpy(np.float64)
        pair_rows.append(
            {
                "well": str(well),
                "candidate_left": left,
                "candidate_right": right,
                "path_rmse_distance": float(np.sqrt(np.mean(difference**2))),
                "path_mean_abs_distance": float(np.mean(np.abs(difference))),
                "path_max_abs_distance": float(np.max(np.abs(difference))),
            }
        )
    meta = {
        "well": str(well),
        "rows": int(len(frame)),
        "prefix_rows": int(result["prefix_rows"]),
        "grid_points": int(len(result["grid"])),
        "rate_states": int(len(result["rates"])),
        "unique_path_count": int(len(result["unique_paths"])),
        "duplicate_joint_path_count": int(TOP_K - len(result["unique_paths"])),
        "backpointer_bytes": int(result["backpointer_bytes"]),
        "prefix_sigma": float(result["prefix_sigma"]),
        "prefix_ir": float(result["prefix_ir"]),
        "log_likelihood": float(result["log_likelihood"]),
        "elapsed_seconds": float(time.time() - started),
        "horizontal_sha256": sha256_path(horizontal_path),
        "typewell_sha256": sha256_path(typewell_path),
    }
    return frame, path_rows, pair_rows, meta


# %% [markdown]
# ## 7. Candidate and oracle audit helpers


# %%
def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def score_prediction(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not valid.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "within_10ft": np.nan,
        }
    error = prediction[valid] - truth[valid]
    return {
        "rows": int(valid.sum()),
        "coverage": float(valid.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "within_10ft": float(np.mean(np.abs(error) <= 10.0)),
    }


def distance_bucket(values: pd.Series, boundaries: list[float]) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    edges = [-np.inf, *[float(value) for value in boundaries], np.inf]
    labels: list[str] = []
    for index in range(len(edges) - 1):
        lower = "neg_inf" if not np.isfinite(edges[index]) else f"{edges[index]:g}"
        upper = "pos_inf" if not np.isfinite(edges[index + 1]) else f"{edges[index + 1]:g}"
        labels.append(f"{lower}_{upper}")
    selected = np.full(len(numeric), "missing", dtype=object)
    for index, label in enumerate(labels):
        selected[(numeric > edges[index]) & (numeric <= edges[index + 1])] = label
    return selected


def compute_direct_metrics(
    frame: pd.DataFrame,
    candidates: tuple[str, ...] = ALL_CANDIDATES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = numeric_array(frame, "true_tvt_readout_only")
    posterior_mean = numeric_array(frame, "posterior_mean")
    overall_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    buckets = distance_bucket(frame["md_since"], [50, 100, 250, 500, 1000])
    mean_rmse = rmse(truth, posterior_mean)
    for candidate in candidates:
        prediction = numeric_array(frame, candidate)
        metric = score_prediction(prediction, truth)
        metric["candidate"] = candidate
        metric["delta_rmse_vs_posterior_mean"] = (
            float(metric["rmse"]) - mean_rmse if np.isfinite(metric["rmse"]) else np.nan
        )
        metric["available_wells"] = int(
            frame.loc[np.isfinite(prediction), "well"].astype(str).nunique()
        )
        overall_rows.append(metric)
        for bucket in sorted(set(buckets)):
            mask = buckets == bucket
            bucket_metric = score_prediction(prediction[mask], truth[mask])
            distance_rows.append(
                {"candidate": candidate, "distance_bucket": str(bucket), **bucket_metric}
            )
        for well, group in frame.groupby("well", sort=False):
            group_prediction = numeric_array(group, candidate)
            group_truth = numeric_array(group, "true_tvt_readout_only")
            group_metric = score_prediction(group_prediction, group_truth)
            if int(group_metric["rows"]) == 0:
                continue
            group_mean = numeric_array(group, "posterior_mean")
            group_metric.update(
                {
                    "metric_kind": "direct_candidate",
                    "candidate": candidate,
                    "well": str(well),
                    "posterior_mean_rmse": rmse(group_truth, group_mean),
                    "delta_rmse_vs_posterior_mean": float(group_metric["rmse"])
                    - rmse(group_truth, group_mean),
                }
            )
            by_well_rows.append(group_metric)
    overall = pd.DataFrame(overall_rows)
    by_well = pd.DataFrame(by_well_rows)
    if not by_well.empty:
        worst = (
            by_well.groupby("candidate", as_index=False)["delta_rmse_vs_posterior_mean"]
            .max()
            .rename(
                columns={"delta_rmse_vs_posterior_mean": "max_well_regression_vs_posterior_mean"}
            )
        )
        overall = overall.merge(worst, on="candidate", how="left", validate="one_to_one")
    return overall, pd.DataFrame(distance_rows), by_well


def compute_by_well_oracle_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        group = group.reset_index(drop=True)
        truth = numeric_array(group, "true_tvt_readout_only")
        for scope, block_rows in (
            ("row", None),
            ("block", 128),
            ("block", 256),
            ("block", 512),
            ("well", None),
        ):
            prediction, _ = oracle_prediction(
                group, ALL_CANDIDATES, scope=scope, block_rows=block_rows
            )
            rows.append(
                {
                    "metric_kind": "oracle_diagnostic",
                    "candidate": (
                        f"oracle_{scope}" if block_rows is None else f"oracle_block_{block_rows}"
                    ),
                    "well": str(well),
                    **score_prediction(prediction, truth),
                    "posterior_mean_rmse": rmse(truth, numeric_array(group, "posterior_mean")),
                    "delta_rmse_vs_posterior_mean": score_prediction(prediction, truth)["rmse"]
                    - rmse(truth, numeric_array(group, "posterior_mean")),
                }
            )
    return pd.DataFrame(rows)


def oracle_prediction(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    scope: str,
    block_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a transient diagnostic oracle; callers must never persist it."""
    truth = numeric_array(frame, "true_tvt_readout_only")
    matrix = frame[list(candidates)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    squared = (matrix - truth[:, None]) ** 2
    squared[~np.isfinite(matrix)] = np.inf
    choice = np.full(len(frame), -1, dtype=np.int16)
    if scope == "row":
        choice = np.argmin(squared, axis=1).astype(np.int16)
    else:
        groups: list[np.ndarray] = []
        for _, group in frame.groupby("well", sort=False):
            indices = group.index.to_numpy(np.int64)
            if scope == "well":
                groups.append(indices)
            elif scope == "block":
                if block_rows is None or block_rows <= 0:
                    raise ValueError("positive block_rows is required for block oracle")
                for start in range(0, len(indices), int(block_rows)):
                    groups.append(indices[start : start + int(block_rows)])
            else:
                raise ValueError(f"unsupported oracle scope={scope}")
        for indices in groups:
            candidate_sse = np.sum(squared[indices], axis=0)
            selected = int(np.argmin(candidate_sse))
            if not np.isfinite(candidate_sse[selected]):
                raise ValueError(f"oracle group has no complete candidate: scope={scope}")
            choice[indices] = selected
    if (choice < 0).any():
        raise RuntimeError(f"oracle failed to assign every row for scope={scope}")
    prediction = matrix[np.arange(len(frame)), choice]
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"oracle selected non-finite prediction for scope={scope}")
    return prediction, choice


def compute_oracle_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    banks = {
        "all_modes": ALL_CANDIDATES,
        "paths_only": PATH_COLUMNS,
        "mean_map_viterbi": ("posterior_mean", "marginal_map", "topk_path_1"),
    }
    truth = numeric_array(frame, "true_tvt_readout_only")
    for bank, configured_candidates in banks.items():
        # A path rank can be absent for some wells after TVT dedup. Keep it in
        # the bank; non-finite values are treated as unavailable for that group.
        candidates = tuple(configured_candidates)
        for scope, block_rows in (
            ("row", None),
            ("block", 128),
            ("block", 256),
            ("block", 512),
            ("well", None),
        ):
            prediction, choice = oracle_prediction(frame, candidates, scope, block_rows)
            metric = score_prediction(prediction, truth)
            counts = np.bincount(choice, minlength=len(candidates))
            rows.append(
                {
                    "bank": bank,
                    "scope": scope if block_rows is None else f"block_{block_rows}",
                    "candidate_count": len(candidates),
                    **metric,
                    "choice_counts_json": json.dumps(
                        {
                            candidate: int(count)
                            for candidate, count in zip(candidates, counts, strict=True)
                        }
                    ),
                }
            )
            del prediction, choice
    return pd.DataFrame(rows)


def compute_unique_best(frame: pd.DataFrame, atol: float) -> pd.DataFrame:
    truth = numeric_array(frame, "true_tvt_readout_only")
    matrix = frame[list(ALL_CANDIDATES)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    absolute_error = np.abs(matrix - truth[:, None])
    absolute_error[~np.isfinite(matrix)] = np.inf
    row_min = np.min(absolute_error, axis=1)
    row_ties = np.abs(absolute_error - row_min[:, None]) <= float(atol)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(ALL_CANDIDATES):
        rows.append(
            {
                "scope": "row",
                "candidate": candidate,
                "units": len(frame),
                "available_units": int(np.isfinite(matrix[:, index]).sum()),
                "unique_best_units": int((row_ties[:, index] & (row_ties.sum(axis=1) == 1)).sum()),
                "tied_best_units": int(row_ties[:, index].sum()),
            }
        )
    for candidate in ALL_CANDIDATES:
        unique_wins = 0
        tied_wins = 0
        available = 0
        for _, group in frame.groupby("well", sort=False):
            candidate_rmse: list[float] = []
            for other in ALL_CANDIDATES:
                metric = score_prediction(
                    numeric_array(group, other), numeric_array(group, "true_tvt_readout_only")
                )
                candidate_rmse.append(float(metric["rmse"]))
            values = np.asarray(candidate_rmse, dtype=np.float64)
            selected_index = ALL_CANDIDATES.index(candidate)
            if not np.isfinite(values[selected_index]):
                continue
            available += 1
            minimum = np.nanmin(values)
            tied = np.isfinite(values) & (np.abs(values - minimum) <= float(atol))
            tied_wins += int(tied[selected_index])
            unique_wins += int(tied[selected_index] and tied.sum() == 1)
        rows.append(
            {
                "scope": "well",
                "candidate": candidate,
                "units": int(frame["well"].nunique()),
                "available_units": available,
                "unique_best_units": unique_wins,
                "tied_best_units": tied_wins,
            }
        )
    output = pd.DataFrame(rows)
    output["unique_best_rate"] = output["unique_best_units"] / output["units"]
    output["tied_best_rate"] = output["tied_best_units"] / output["units"]
    return output


def compute_hidden_like_metrics(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, Path | None]:
    hidden = get_nested(config, "data.hidden_like") or {}
    if not bool(hidden.get("enabled", False)):
        return pd.DataFrame(), None
    candidates = [str(value) for value in hidden.get("fold_assignment_candidates", [])]
    filename = Path(candidates[0]).name
    path = resolve_existing(filename, candidates)
    assignments = pd.read_csv(path, dtype={"well_id": str})
    rows: list[dict[str, Any]] = []
    truth = numeric_array(frame, "true_tvt_readout_only")
    for subgroup, role_column in (hidden.get("valid_role_columns") or {}).items():
        if role_column not in assignments.columns:
            raise ValueError(f"hidden-like assignment missing {role_column}")
        wells = set(
            assignments.loc[assignments[role_column].astype(str) == "valid", "well_id"].astype(str)
        )
        mask = frame["well"].astype(str).isin(wells).to_numpy()
        if not mask.any():
            raise ValueError(f"hidden-like subgroup {subgroup} selected zero rows")
        for candidate in ALL_CANDIDATES:
            rows.append(
                {
                    "subgroup": str(subgroup),
                    "candidate": candidate,
                    **score_prediction(numeric_array(frame, candidate)[mask], truth[mask]),
                }
            )
    return pd.DataFrame(rows), path


def compute_focus_well(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well in get_nested(config, "audit.focus_wells") or []:
        group = frame.loc[frame["well"].astype(str) == str(well)].copy()
        if group.empty:
            rows.append({"well": str(well), "candidate": None, "status": "not_found"})
            continue
        truth = numeric_array(group, "true_tvt_readout_only")
        for candidate in ALL_CANDIDATES:
            rows.append(
                {
                    "well": str(well),
                    "candidate": candidate,
                    "status": "ok",
                    **score_prediction(numeric_array(group, candidate), truth),
                }
            )
        for scope, block_rows in (
            ("row", None),
            ("block", 128),
            ("block", 256),
            ("block", 512),
            ("well", None),
        ):
            prediction, _ = oracle_prediction(
                group.reset_index(drop=True), ALL_CANDIDATES, scope, block_rows
            )
            rows.append(
                {
                    "well": str(well),
                    "candidate": (
                        f"oracle_{scope}" if block_rows is None else f"oracle_block_{block_rows}"
                    ),
                    "status": "diagnostic_only",
                    **score_prediction(prediction, truth),
                }
            )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 8. Shard generation and aggregate orchestration


# %%
def configured_shard_spec(config: dict[str, Any], shard_index: int) -> dict[str, Any]:
    matches = [
        spec
        for spec in get_nested(config, "data.shard_outputs") or []
        if int(spec.get("shard_index", -1)) == int(shard_index)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one shard spec for index={shard_index}; found={len(matches)}")
    return matches[0]


def run_shard_generation(config: dict[str, Any], shard_index: int) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp270 shard generation must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 "
            "is reserved for an explicitly approved local smoke run."
        )
    validate_scientific_contract(config)
    shard_count = int(get_nested(config, "execution.shard_count") or 0)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid shard index {shard_index} for shard_count={shard_count}")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for exact-HMM shard generation")
    requested_threads = int(get_nested(config, "execution.numba_num_threads") or 1)
    set_num_threads(requested_threads)
    data_dir = train_data_dir(config)
    wells = list_well_ids(data_dir)
    selected = [well for well in wells if stable_well_shard(well, shard_count) == shard_index]
    configured_max = get_nested(config, "execution.max_wells_per_shard")
    environment_max = int(os.environ.get("EXPERIMENT_MAX_WELLS", "0") or "0")
    max_wells = environment_max or (
        int(configured_max) if configured_max is not None else None
    )
    if max_wells is not None:
        selected = selected[:max_wells]
    if not selected:
        raise ValueError(f"no wells selected for shard {shard_index}")

    started = time.time()
    path_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    progress_every = int(get_nested(config, "execution.progress_every_wells") or 1)
    full_shard = max_wells is None
    spec = configured_shard_spec(config, shard_index)
    artifacts = artifact_dir()
    prefix = f"{OUTPUT_PREFIX}_shard{shard_index}"
    paths = {
        "candidates": artifacts / f"{prefix}_candidates.csv.gz",
        "candidate_schema": artifacts / f"{prefix}_candidate_schema.csv",
        "decoder_manifest": artifacts / f"{prefix}_decoder_manifest.json",
        "path_diagnostics": artifacts / f"{prefix}_path_diagnostics.csv",
        "pairwise_path_distance": artifacts / f"{prefix}_pairwise_path_distance.csv",
        "well_manifest": artifacts / f"{prefix}_well_manifest.csv",
        "input_manifest": artifacts / f"{prefix}_input_manifest.csv",
        "summary": artifacts / f"{prefix}_summary.json",
    }
    candidate_binary_path = artifacts / f".{prefix}_candidate_matrix.float32.part"
    row_idx_binary_path = artifacts / f".{prefix}_row_idx.int64.part"
    flush_every = int(get_nested(config, "execution.stream_flush_every_wells") or 10)
    compresslevel = int(get_nested(config, "execution.gzip_compresslevel") or 1)
    gzip_mtime = int(get_nested(config, "execution.gzip_mtime") or 0)
    hash_chunk_bytes = int(
        get_nested(config, "execution.binary_hash_chunk_bytes") or 8 * 1024 * 1024
    )
    expected_columns: list[str] | None = None
    expected_dtypes: list[str] | None = None
    rows_written = 0
    log_stage(
        f"shard{shard_index}_stream_start",
        started,
        selected_wells=len(selected),
        expected_rows=int(spec["expected_rows"]) if full_shard else None,
    )
    with paths["candidates"].open("wb") as raw_candidate_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_candidate_file,
            compresslevel=compresslevel,
            mtime=gzip_mtime,
        ) as gzip_candidate_file:
            with io.TextIOWrapper(
                gzip_candidate_file, encoding="utf-8", newline=""
            ) as candidate_text_file:
                with candidate_binary_path.open("wb") as candidate_binary_file:
                    with row_idx_binary_path.open("wb") as row_idx_binary_file:
                        for index, well in enumerate(selected, start=1):
                            print(
                                f"[exp270 shard{shard_index}] {index}/{len(selected)} "
                                f"well={well}",
                                flush=True,
                            )
                            frame, well_paths, well_pairs, meta = build_candidate_rows_for_well(
                                well, data_dir, config
                            )
                            columns = list(frame.columns)
                            dtypes = [str(frame[column].dtype) for column in columns]
                            if expected_columns is None:
                                expected_columns = columns
                                expected_dtypes = dtypes
                            elif columns != expected_columns or dtypes != expected_dtypes:
                                raise ValueError(
                                    f"shard {shard_index} candidate schema changed at well={well}"
                                )
                            if frame["id"].duplicated().any():
                                raise ValueError(f"duplicate generated id for well={well}")
                            if frame["well"].astype(str).nunique() != 1 or str(
                                frame["well"].iloc[0]
                            ) != str(well):
                                raise ValueError(f"candidate well label mismatch for well={well}")
                            row_idx = frame["row_idx"].to_numpy(np.int64)
                            if len(row_idx) > 1 and bool((np.diff(row_idx) <= 0).any()):
                                raise ValueError(
                                    f"candidate row_idx is not strictly ordered for well={well}"
                                )
                            fixed_values = frame[list(FIXED_CANDIDATES)].to_numpy(np.float64)
                            if not np.isfinite(fixed_values).all():
                                raise RuntimeError(
                                    f"shard {shard_index} fixed candidates contain "
                                    "non-finite values"
                                )
                            frame.to_csv(
                                candidate_text_file,
                                index=False,
                                header=rows_written == 0,
                            )
                            np.ascontiguousarray(
                                frame[list(ALL_CANDIDATES)].to_numpy(np.float32)
                            ).tofile(candidate_binary_file)
                            np.ascontiguousarray(row_idx).tofile(row_idx_binary_file)
                            rows_written += len(frame)
                            path_rows.extend(well_paths)
                            pair_rows.extend(well_pairs)
                            meta.update(
                                {
                                    "progress_index": index,
                                    "cumulative_rows_written": rows_written,
                                    **memory_usage_mb(),
                                }
                            )
                            well_rows.append(meta)
                            if index == 1 or index % progress_every == 0 or index == len(selected):
                                print(json.dumps(to_jsonable(meta), sort_keys=True), flush=True)
                            if index % flush_every == 0 or index == len(selected):
                                candidate_text_file.flush()
                                gzip_candidate_file.flush()
                                candidate_binary_file.flush()
                                row_idx_binary_file.flush()
                            del frame, fixed_values, row_idx, well_paths, well_pairs
                            gc.collect()
    if expected_columns is None or expected_dtypes is None:
        raise RuntimeError(f"shard {shard_index} did not generate a candidate schema")
    log_stage(
        f"shard{shard_index}_hmm_and_stream_complete",
        started,
        rows=rows_written,
        wells=len(well_rows),
    )
    if full_shard and (
        rows_written != int(spec["expected_rows"])
        or len(well_rows) != int(spec["expected_wells"])
    ):
        raise ValueError(
            f"shard {shard_index} coverage mismatch rows={rows_written}/{spec['expected_rows']} "
            f"wells={len(well_rows)}/{spec['expected_wells']}"
        )
    if len({str(row["well"]) for row in well_rows}) != len(well_rows):
        raise RuntimeError(f"shard {shard_index} well manifest contains duplicate wells")
    if any(
        stable_well_shard(str(row["well"]), shard_count) != shard_index for row in well_rows
    ):
        raise RuntimeError(f"shard {shard_index} contains a well assigned to another shard")
    log_stage(f"shard{shard_index}_parity_start", started, rows=rows_written)
    parity_chunk_rows = int(get_nested(config, "execution.parity_chunksize_rows") or 100000)
    parity, control_path, control_decompressed_sha = validate_posterior_mean_parity_batches(
        iter_candidate_csv_batches(paths["candidates"], parity_chunk_rows),
        config,
        expected_rows=rows_written,
        selected_wells=set(selected),
    )
    log_stage(
        f"shard{shard_index}_parity_complete",
        started,
        parity_max_abs_diff_ft=parity["max_abs_diff_ft"],
    )
    pd.DataFrame(
        {
            "column_index": np.arange(len(expected_columns), dtype=np.int32),
            "column": expected_columns,
            "dtype": expected_dtypes,
        }
    ).to_csv(paths["candidate_schema"], index=False)
    path_frame = pd.DataFrame(path_rows).sort_values(["well", "joint_rank"])
    pair_frame = pd.DataFrame(pair_rows).sort_values(
        ["well", "candidate_left", "candidate_right"]
    )
    well_frame = pd.DataFrame(well_rows).sort_values("well")
    path_frame.to_csv(paths["path_diagnostics"], index=False)
    pair_frame.to_csv(paths["pairwise_path_distance"], index=False)
    well_frame.to_csv(paths["well_manifest"], index=False)
    input_rows = well_frame[["well", "horizontal_sha256", "typewell_sha256"]].to_dict(
        "records"
    )
    input_rows.append(
        {
            "well": "__exp209_control__",
            "horizontal_sha256": sha256_path(control_path),
            "typewell_sha256": control_decompressed_sha,
        }
    )
    pd.DataFrame(input_rows).to_csv(paths["input_manifest"], index=False)
    decoder_manifest = {
        "hmm": get_nested(config, "model.hmm"),
        "decoder": get_nested(config, "model.decoder"),
        "candidate_bank": get_nested(config, "candidate_bank"),
        "shard_policy": get_nested(config, "execution.shard_policy"),
    }
    write_json(paths["decoder_manifest"], decoder_manifest)
    log_stage(f"shard{shard_index}_prediction_sha_start", started, rows=rows_written)
    prediction_sha = array_bundle_sha256_from_binary_parts(
        candidate_path=candidate_binary_path,
        row_idx_path=row_idx_binary_path,
        rows=rows_written,
        candidate_count=len(ALL_CANDIDATES),
        chunk_bytes=hash_chunk_bytes,
    )
    candidate_binary_path.unlink()
    row_idx_binary_path.unlink()
    log_stage(f"shard{shard_index}_artifact_sha_start", started, rows=rows_written)
    candidates_raw_sha = sha256_path(paths["candidates"])
    candidates_decompressed_sha = sha256_gzip_decompressed(paths["candidates"])
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "shard_generation_completed",
        "run_kind": f"shard{shard_index}",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_policy": get_nested(config, "execution.shard_policy"),
        "rows": rows_written,
        "wells": len(well_rows),
        "full_shard": full_shard,
        "posterior_mean_parity": parity,
        "unique_path_count": {
            "mean": float(well_frame["unique_path_count"].mean()),
            "minimum": int(well_frame["unique_path_count"].min()),
            "wells_below_five": int((well_frame["unique_path_count"] < TOP_K).sum()),
        },
        "active_hmm_variants": 1,
        "hmm_well_runs": len(selected),
        "lightgbm_configs": 0,
        "folds": 0,
        "boosters": 0,
        "gpu": False,
        "inference": False,
        "submission": False,
        "numba_threads": requested_threads,
        "elapsed_seconds": float(time.time() - started),
        "memory": memory_usage_mb(),
        "streaming": {
            "candidate_write": "well_ordered_single_gzip_stream",
            "gzip_compresslevel": compresslevel,
            "gzip_mtime": gzip_mtime,
            "flush_every_wells": flush_every,
            "parity_chunk_rows": parity_chunk_rows,
            "prediction_sha_binary_chunk_bytes": hash_chunk_bytes,
            "full_dataframe_concat": False,
            "object_id_setdiff": False,
        },
        "prediction_content_sha256": prediction_sha,
        "decoder_manifest_sha256": mapping_sha256(decoder_manifest),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "sha256": {
            "candidates_raw_gzip": candidates_raw_sha,
            "candidates_decompressed": candidates_decompressed_sha,
            "candidate_schema": sha256_path(paths["candidate_schema"]),
            "decoder_manifest": sha256_path(paths["decoder_manifest"]),
            "path_diagnostics": sha256_path(paths["path_diagnostics"]),
            "pairwise_path_distance": sha256_path(paths["pairwise_path_distance"]),
            "well_manifest": sha256_path(paths["well_manifest"]),
            "input_manifest": sha256_path(paths["input_manifest"]),
        },
    }
    write_json(paths["summary"], summary)
    summary["sha256"]["summary"] = sha256_path(paths["summary"])
    write_json(paths["summary"], summary)
    write_json(
        artifacts.parent / "metrics.json",
        {
            "experiment": EXPERIMENT_NAME,
            "status": "shard_generation_completed",
            "run_kind": f"shard{shard_index}",
            "rows": rows_written,
            "wells": len(well_rows),
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "posterior_mean_parity": parity,
            "prediction_content_sha256": prediction_sha,
        },
    )
    log_stage(f"shard{shard_index}_complete", started, rows=rows_written, wells=len(well_rows))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def required_sha(spec: dict[str, Any], key: str, label: str) -> str:
    value = str(spec.get(key) or "")
    if len(value) != 64:
        raise ValueError(f"{label} requires a fixed 64-character {key} before aggregate")
    return value


def load_shards(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    started = time.time()
    frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    well_frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    expected_columns: list[str] | None = None
    shard_count = int(get_nested(config, "execution.shard_count") or 0)
    for spec in get_nested(config, "data.shard_outputs") or []:
        shard_index = int(spec["shard_index"])
        path = resolve_existing(str(spec["filename"]), [str(v) for v in spec["candidates"]])
        raw_sha = sha256_path(path)
        decompressed_sha = sha256_gzip_decompressed(path)
        if raw_sha != required_sha(spec, "expected_raw_sha256", f"shard{shard_index}"):
            raise ValueError(f"shard{shard_index} raw SHA mismatch: {raw_sha}")
        if decompressed_sha != required_sha(
            spec, "expected_decompressed_sha256", f"shard{shard_index}"
        ):
            raise ValueError(f"shard{shard_index} decompressed SHA mismatch: {decompressed_sha}")
        frame = pd.read_csv(path, dtype={"id": str, "well": str})
        wells = int(frame["well"].nunique())
        if len(frame) != int(spec["expected_rows"]) or wells != int(spec["expected_wells"]):
            raise ValueError(
                f"shard{shard_index} coverage mismatch rows={len(frame)} wells={wells}"
            )
        if frame["id"].duplicated().any():
            raise ValueError(f"shard{shard_index} contains duplicate ids")
        if any(
            stable_well_shard(well, shard_count) != shard_index
            for well in frame["well"].astype(str).unique()
        ):
            raise ValueError(f"shard{shard_index} stable-hash assignment mismatch")
        if expected_columns is None:
            expected_columns = list(frame.columns)
        elif list(frame.columns) != expected_columns:
            raise ValueError(f"shard{shard_index} candidate schema/order mismatch")

        sidecars: dict[str, Path] = {}
        for role, filename_key in (
            ("path_diagnostics", "path_diagnostics_filename"),
            ("pairwise_path_distance", "pairwise_filename"),
            ("well_manifest", "well_manifest_filename"),
        ):
            filename = str(spec[filename_key])
            sidecar = resolve_existing(filename, [str(path.parent / filename)])
            expected_key = f"expected_{role}_sha256"
            actual_sha = sha256_path(sidecar)
            if actual_sha != required_sha(spec, expected_key, f"shard{shard_index} {role}"):
                raise ValueError(f"shard{shard_index} {role} SHA mismatch: {actual_sha}")
            sidecars[role] = sidecar
        path_frame = pd.read_csv(sidecars["path_diagnostics"], dtype={"well": str})
        pair_frame = pd.read_csv(sidecars["pairwise_path_distance"], dtype={"well": str})
        well_frame = pd.read_csv(sidecars["well_manifest"], dtype={"well": str})
        if int(well_frame["well"].nunique()) != int(spec["expected_wells"]):
            raise ValueError(f"shard{shard_index} well manifest coverage mismatch")
        frames.append(frame)
        path_frames.append(path_frame)
        pair_frames.append(pair_frame)
        well_frames.append(well_frame)
        manifest.append(
            {
                "shard_index": shard_index,
                "path": str(path),
                "bytes": path.stat().st_size,
                "raw_sha256": raw_sha,
                "decompressed_sha256": decompressed_sha,
                "path_diagnostics_sha256": sha256_path(sidecars["path_diagnostics"]),
                "pairwise_path_distance_sha256": sha256_path(
                    sidecars["pairwise_path_distance"]
                ),
                "well_manifest_sha256": sha256_path(sidecars["well_manifest"]),
                "rows": len(frame),
                "wells": wells,
            }
        )
        log_stage(
            "aggregate_load_shard",
            started,
            shard_index=shard_index,
            rows=len(frame),
            wells=wells,
        )
    if len(frames) != shard_count:
        raise ValueError("all configured shards must be present")
    generated = pd.concat(frames, ignore_index=True, copy=False)
    del frames
    gc.collect()
    if generated["id"].duplicated().any():
        raise ValueError("shard union contains duplicate ids")
    generated = generated.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    log_stage(
        "aggregate_load_and_sort_complete",
        started,
        rows=len(generated),
        wells=int(generated["well"].nunique()),
    )
    return (
        generated,
        pd.concat(path_frames, ignore_index=True),
        pd.concat(pair_frames, ignore_index=True),
        pd.concat(well_frames, ignore_index=True),
        sorted(manifest, key=lambda item: int(item["shard_index"])),
    )


def run_aggregate_from_parts(
    config: dict[str, Any],
    generated: pd.DataFrame,
    path_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    well_rows: pd.DataFrame,
    shard_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_scientific_contract(config)
    requested_threads = int(get_nested(config, "execution.numba_num_threads") or 1)
    started = time.time()
    generated = generated.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    if generated["id"].duplicated().any():
        raise RuntimeError("aggregate generated frame contains duplicate ids")

    is_full_run = True
    expected_rows = int(get_nested(config, "validation.expected_rows") or 0)
    expected_wells = int(get_nested(config, "validation.expected_wells") or 0)
    if is_full_run and (
        len(generated) != expected_rows or generated["well"].nunique() != expected_wells
    ):
        raise ValueError(
            f"coverage mismatch rows={len(generated)}/{expected_rows} "
            f"wells={generated['well'].nunique()}/{expected_wells}"
        )

    parity_chunk_rows = int(get_nested(config, "execution.parity_chunksize_rows") or 100000)
    log_stage("aggregate_parity_start", started, rows=len(generated))
    parity, control_path, control_decompressed_sha = validate_posterior_mean_parity_batches(
        iter_frame_batches(
            generated,
            ["id", "well", "posterior_mean"],
            parity_chunk_rows,
        ),
        config,
        expected_rows=len(generated),
        selected_wells=None,
    )
    log_stage(
        "aggregate_parity_complete",
        started,
        parity_max_abs_diff_ft=parity["max_abs_diff_ft"],
    )

    log_stage("aggregate_metrics_start", started, rows=len(generated))
    overall, distance, by_well = compute_direct_metrics(generated)
    by_well_oracle = compute_by_well_oracle_metrics(generated)
    by_well = pd.concat([by_well, by_well_oracle], ignore_index=True, sort=False)
    oracle = compute_oracle_metrics(generated)
    unique_best = compute_unique_best(
        generated, float(get_nested(config, "audit.unique_best_tie_atol_ft") or 0.0)
    )
    hidden, hidden_path = compute_hidden_like_metrics(generated, config)
    focus = compute_focus_well(generated, config)
    path_diagnostics = path_rows.sort_values(["well", "joint_rank"]).reset_index(drop=True)
    pairwise = pair_rows.sort_values(
        ["well", "candidate_left", "candidate_right"]
    ).reset_index(drop=True)
    well_manifest = well_rows.sort_values("well").reset_index(drop=True)
    log_stage("aggregate_metrics_complete", started, rows=len(generated))

    forbidden_columns = [
        column
        for column in generated.columns
        if "oracle" in column.lower() or "selector" in column.lower() or "blend" in column.lower()
    ]
    if forbidden_columns:
        raise RuntimeError(f"forbidden deployable columns found: {forbidden_columns}")

    artifacts = artifact_dir()
    paths = {
        "candidates": artifacts / f"{OUTPUT_PREFIX}_candidates.csv.gz",
        "candidate_schema": artifacts / f"{OUTPUT_PREFIX}_candidate_schema.csv",
        "decoder_manifest": artifacts / f"{OUTPUT_PREFIX}_decoder_manifest.json",
        "candidate_metrics": artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv",
        "distance_bucket_metrics": artifacts / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv",
        "hidden_like_metrics": artifacts / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv",
        "by_well": artifacts / f"{OUTPUT_PREFIX}_by_well.csv",
        "path_diagnostics": artifacts / f"{OUTPUT_PREFIX}_path_diagnostics.csv",
        "pairwise_path_distance": artifacts / f"{OUTPUT_PREFIX}_pairwise_path_distance.csv",
        "unique_best": artifacts / f"{OUTPUT_PREFIX}_unique_best.csv",
        "oracle_scope_metrics": artifacts / f"{OUTPUT_PREFIX}_oracle_scope_metrics.csv",
        "focus_well": artifacts / f"{OUTPUT_PREFIX}_focus_well.csv",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "summary": artifacts / f"{OUTPUT_PREFIX}_summary.json",
    }
    frame_write_chunk_rows = int(
        get_nested(config, "execution.frame_write_chunksize_rows") or 100000
    )
    compresslevel = int(get_nested(config, "execution.gzip_compresslevel") or 1)
    gzip_mtime = int(get_nested(config, "execution.gzip_mtime") or 0)
    log_stage("aggregate_candidate_write_start", started, rows=len(generated))
    write_dataframe_gzip_deterministic(
        paths["candidates"],
        generated,
        chunk_rows=frame_write_chunk_rows,
        compresslevel=compresslevel,
        mtime=gzip_mtime,
    )
    pd.DataFrame(
        {
            "column_index": np.arange(len(generated.columns), dtype=np.int32),
            "column": generated.columns,
            "dtype": [str(generated[column].dtype) for column in generated.columns],
        }
    ).to_csv(paths["candidate_schema"], index=False)
    overall.to_csv(paths["candidate_metrics"], index=False)
    distance.to_csv(paths["distance_bucket_metrics"], index=False)
    hidden.to_csv(paths["hidden_like_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    path_diagnostics.to_csv(paths["path_diagnostics"], index=False)
    pairwise.to_csv(paths["pairwise_path_distance"], index=False)
    unique_best.to_csv(paths["unique_best"], index=False)
    oracle.to_csv(paths["oracle_scope_metrics"], index=False)
    focus.to_csv(paths["focus_well"], index=False)

    input_rows = well_manifest[["well", "horizontal_sha256", "typewell_sha256"]].to_dict("records")
    for item in shard_manifest:
        input_rows.append(
            {
                "well": f"__shard{int(item['shard_index'])}__",
                "horizontal_sha256": item["raw_sha256"],
                "typewell_sha256": item["decompressed_sha256"],
            }
        )
    input_rows.append(
        {
            "well": "__exp209_control__",
            "horizontal_sha256": sha256_path(control_path),
            "typewell_sha256": control_decompressed_sha,
        }
    )
    if hidden_path is not None:
        input_rows.append(
            {
                "well": "__hidden_like_assignments__",
                "horizontal_sha256": sha256_path(hidden_path),
                "typewell_sha256": None,
            }
        )
    input_manifest = pd.DataFrame(input_rows)
    input_manifest.to_csv(paths["input_manifest"], index=False)

    log_stage("aggregate_prediction_sha_start", started, rows=len(generated))
    prediction_sha = array_bundle_sha256_from_frame(
        generated,
        chunk_rows=frame_write_chunk_rows,
    )
    decoder_manifest = {
        "hmm": get_nested(config, "model.hmm"),
        "decoder": get_nested(config, "model.decoder"),
        "candidate_bank": get_nested(config, "candidate_bank"),
    }
    decoder_manifest_sha = mapping_sha256(decoder_manifest)
    write_json(paths["decoder_manifest"], decoder_manifest)
    posterior_mean_row = overall.loc[overall["candidate"] == "posterior_mean"].iloc[0].to_dict()
    full_coverage = overall.loc[np.isclose(overall["coverage"], 1.0)].copy()
    if full_coverage.empty:
        raise RuntimeError("no full-coverage direct candidate was generated")
    best_direct = (
        full_coverage.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    direct_by_well = by_well.loc[by_well["metric_kind"].astype(str) == "direct_candidate"].copy()
    worst_well_by_candidate = (
        direct_by_well.sort_values(
            ["candidate", "delta_rmse_vs_posterior_mean", "well"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("candidate", as_index=False, sort=False)
        .head(1)
        .sort_values("candidate")
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_mode_audit_pending_review",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "numba": getattr(numba, "__version__", None),
            "numba_threads": requested_threads,
            "kaggle_kernel_version": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            "elapsed_seconds": float(time.time() - started),
            **memory_usage_mb(),
        },
        "rows": len(generated),
        "wells": int(generated["well"].nunique()),
        "full_run": is_full_run,
        "run_kind": "aggregate",
        "shard_policy": get_nested(config, "execution.shard_policy"),
        "shard_inputs": shard_manifest,
        "joint_top_k": TOP_K,
        "tvt_path_dedup_backfill": False,
        "posterior_mean_parity": parity,
        "posterior_mean": posterior_mean_row,
        "best_full_coverage_direct_candidate_target_side_diagnostic_only": best_direct,
        "candidate_metrics": overall.sort_values("candidate").to_dict("records"),
        "oracle_scope_metrics": oracle.sort_values(["bank", "scope"]).to_dict("records"),
        "hidden_like_metrics": hidden.sort_values(["subgroup", "candidate"]).to_dict("records"),
        "focus_well_metrics": focus.sort_values(["well", "candidate"]).to_dict("records"),
        "unique_best_metrics": unique_best.sort_values(["scope", "candidate"]).to_dict("records"),
        "worst_well_by_candidate": worst_well_by_candidate.to_dict("records"),
        "unique_path_count": {
            "mean": float(well_manifest["unique_path_count"].mean()),
            "minimum": int(well_manifest["unique_path_count"].min()),
            "wells_below_five": int((well_manifest["unique_path_count"] < TOP_K).sum()),
        },
        "oracle_prediction_persisted": False,
        "selector_persisted": False,
        "inference_enabled": False,
        "submission_created": False,
        "prediction_content_sha256": prediction_sha,
        "streaming": {
            "parity_alignment": "linear_ordered_id_well_chunks",
            "parity_chunk_rows": parity_chunk_rows,
            "candidate_write_chunk_rows": frame_write_chunk_rows,
            "gzip_compresslevel": compresslevel,
            "gzip_mtime": gzip_mtime,
            "prediction_sha_chunked": True,
            "object_id_setdiff": False,
        },
        "decoder_manifest_sha256": decoder_manifest_sha,
        "candidate_schema_sha256": sha256_path(paths["candidate_schema"]),
        "candidate_gzip_raw_sha256": sha256_path(paths["candidates"]),
        "candidate_gzip_decompressed_sha256": sha256_gzip_decompressed(paths["candidates"]),
        "input_manifest_sha256": sha256_path(paths["input_manifest"]),
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    write_json(paths["summary"], summary)
    artifact_sha = {
        key: sha256_path(path) for key, path in paths.items() if key != "summary" and path.exists()
    }
    summary["artifact_sha256"] = artifact_sha
    write_json(paths["summary"], summary)
    metrics_path = artifacts.parent / "metrics.json"
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": "completed_train_side_mode_audit_pending_review",
            "metric": "rmse_tvt",
            "cv": posterior_mean_row["rmse"],
            "public_lb": None,
            "private_lb": None,
            "rows": len(generated),
            "wells": int(generated["well"].nunique()),
            "posterior_mean_parity": parity,
            "best_full_coverage_direct_candidate_target_side_diagnostic_only": best_direct,
            "candidate_metrics": overall.sort_values("candidate").to_dict("records"),
            "oracle_scope_metrics": oracle.sort_values(["bank", "scope"]).to_dict("records"),
            "prediction_content_sha256": prediction_sha,
            "decoder_manifest_sha256": decoder_manifest_sha,
            "summary": str(paths["summary"]),
        },
    )
    log_stage(
        "aggregate_complete",
        started,
        rows=len(generated),
        wells=int(generated["well"].nunique()),
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def run_aggregate(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp270 aggregation must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 "
            "is reserved for an explicitly approved local smoke run."
        )
    validate_scientific_contract(config)
    generated, path_rows, pair_rows, well_rows, shard_manifest = load_shards(config)
    expected_rows = int(get_nested(config, "validation.expected_rows") or 0)
    expected_wells = int(get_nested(config, "validation.expected_wells") or 0)
    if len(generated) != expected_rows or generated["well"].nunique() != expected_wells:
        raise ValueError(
            f"aggregate coverage mismatch rows={len(generated)}/{expected_rows} "
            f"wells={generated['well'].nunique()}/{expected_wells}"
        )
    return run_aggregate_from_parts(
        config,
        generated,
        path_rows,
        pair_rows,
        well_rows,
        shard_manifest,
    )


# %% [markdown]
# ## 9. Setup and input preflight

# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "run_kind": RUN_KIND_OVERRIDE,
                "active_hmm_variants": get_nested(CONFIG, "execution.active_hmm_variants"),
                "total_hmm_well_runs": get_nested(CONFIG, "execution.total_hmm_well_runs"),
                "well_shards": get_nested(CONFIG, "execution.shard_count"),
                "lightgbm_configs": get_nested(CONFIG, "execution.lightgbm_config_count"),
                "folds": get_nested(CONFIG, "execution.fold_count"),
                "boosters": get_nested(CONFIG, "execution.total_boosters"),
                "top_k": get_nested(CONFIG, "model.decoder.joint_top_k"),
                "deduplicate_by": get_nested(CONFIG, "model.decoder.deduplicate_by"),
                "oracle_blocks": get_nested(CONFIG, "audit.oracle_block_rows"),
                "gpu": get_nested(CONFIG, "execution.gpu"),
                "inference": get_nested(CONFIG, "execution.inference"),
                "submission": get_nested(CONFIG, "execution.submission"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

# %%
if EXECUTE_NOTEBOOK:
    if RUN_KIND_OVERRIDE in {"shard0", "shard1"}:
        TRAIN_DATA = train_data_dir(CONFIG)
        WELL_IDS = list_well_ids(TRAIN_DATA)
        if not WELL_IDS:
            raise FileNotFoundError(f"raw train well pairs not found: {TRAIN_DATA}")
        CONTROL_SPEC = get_nested(CONFIG, "data.exp209_hmm_control") or {}
        CONTROL_PATH = resolve_existing(
            str(CONTROL_SPEC["filename"]),
            [str(value) for value in CONTROL_SPEC["candidates"]],
        )
        print(
            json.dumps(
                {
                    "train_data_dir": str(TRAIN_DATA),
                    "raw_wells": len(WELL_IDS),
                    "first_wells": WELL_IDS[:5],
                    "exp209_control": str(CONTROL_PATH),
                    "exp209_control_bytes": CONTROL_PATH.stat().st_size,
                    "numba_available": NUMBA_AVAILABLE,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    elif RUN_KIND_OVERRIDE == "aggregate":
        print(
            "Aggregate preflight resolves two fixed-SHA shard caches, "
            "the exp209 posterior-mean control, and exp115 hidden-like folds.",
            flush=True,
        )
    else:
        raise ValueError(f"unsupported RUN_KIND_OVERRIDE={RUN_KIND_OVERRIDE}")


# %% [markdown]
# ## 10. Generate a shard or aggregate the audit

# %%
if EXECUTE_NOTEBOOK:
    if RUN_KIND_OVERRIDE == "shard0":
        SUMMARY = run_shard_generation(CONFIG, shard_index=0)
    elif RUN_KIND_OVERRIDE == "shard1":
        SUMMARY = run_shard_generation(CONFIG, shard_index=1)
    else:
        SUMMARY = run_aggregate(CONFIG)


# %% [markdown]
# ## 11. Metrics and artifact summary

# %%
if EXECUTE_NOTEBOOK:
    print(json.dumps(to_jsonable(SUMMARY), indent=2, sort_keys=True), flush=True)

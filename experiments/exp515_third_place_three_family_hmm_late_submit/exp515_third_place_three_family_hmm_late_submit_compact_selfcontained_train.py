# %% [markdown]
# # exp515 — 3rd-place three-family HMM LATE SUBMIT reconstruction (train)
#
# This is a post-competition, writeup-based reconstruction audit.  The original
# source code was not public.  All published HMM mechanisms are represented,
# while unreported lattice/transition values are frozen in `config.yaml` before
# OOF or leaderboard results are observed.

# %% [markdown]
# ## Contents
# 1. Imports and execution policy
# 2. Configuration, paths, SHA, and frozen-contract helpers
# 3. Fold-safe sibling-reference construction
# 4. Joint TVT/rate/bias/reference exact forward-backward kernel
# 5. Three HMM families and physical projection
# 6. Fold-safe OOF orchestration and prediction freeze
# 7. Metrics, diagnostics, and generated artifacts
# 8. Setup preview and Kaggle CPU execution

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def prange(*args: Any) -> range:
        return range(*args)

    def set_num_threads(_: int) -> None:
        return None

    def get_num_threads() -> int | None:
        return None

    def njit(*args: Any, **_: Any) -> Any:
        if args and callable(args[0]):
            return args[0]

        def decorator(func: Any) -> Any:
            return func

        return decorator


EXPERIMENT_NAME = "exp515_third_place_three_family_hmm_late_submit"
OUTPUT_PREFIX = EXPERIMENT_NAME
FAMILY_ORDER = (
    "exp417_base_reconstruction",
    "local_dtw_reconstruction",
    "fine_bin_reconstruction",
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP515_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Configuration, paths, SHA, and frozen-contract helpers

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


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


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
        if (candidate / "project.yml").is_file():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp515 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    override = os.environ.get("EXP515_OUTPUT_DIR")
    if override:
        path = Path(override)
    elif KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    override = os.environ.get("EXP515_OUTPUT_DIR")
    if override:
        return Path(override) / "metrics.json"
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def resolve_competition_root(config: dict[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction",
        )
        for candidate in fixed:
            if (candidate / "train").is_dir() and (candidate / "sample_submission.csv").is_file():
                return candidate
        for sample in sorted(KAGGLE_INPUT_ROOT.glob("**/sample_submission.csv")):
            if (sample.parent / "train").is_dir():
                return sample.parent
    local = project_root() / str(get_nested(config, "data.raw_dir") or "data/raw")
    if (local / "train").is_dir():
        return local
    raise FileNotFoundError("competition root with train/ and sample_submission.csv was not found")


def resolve_group_assignment(config: dict[str, Any]) -> Path:
    filename = str(get_nested(config, "data.typewell_group_assignment.filename"))
    candidates = list(get_nested(config, "data.typewell_group_assignment.candidates") or [])
    root = project_root()
    for raw in candidates:
        path = Path(str(raw))
        for candidate in (path, root / path, Path.cwd() / path):
            if candidate.is_file():
                return candidate
    if KAGGLE_INPUT_ROOT.exists():
        found = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        if found:
            return found[0]
    raise FileNotFoundError(f"could not resolve {filename}")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(column.encode())
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            array = np.ascontiguousarray(values.to_numpy())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        else:
            for value in values.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def inspect_gzip_csv(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    newlines = 0
    last = b""
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
            newlines += chunk.count(b"\n")
            if chunk:
                last = chunk[-1:]
    lines = newlines + int(bool(last) and last != b"\n")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": digest.hexdigest(),
        "data_rows": max(0, lines - 1),
    }


def list_wells(data_dir: Path) -> list[str]:
    suffix = "__horizontal_well.csv"
    wells = []
    for path in sorted(data_dir.glob(f"*{suffix}")):
        well = path.name.removesuffix(suffix)
        if (data_dir / f"{well}__typewell.csv").is_file():
            wells.append(well)
    return wells


def load_horizontal(data_dir: Path, well: str, *, include_truth: bool) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"{well}__horizontal_well.csv")
    required = {"MD", "Z", "GR", "TVT_input"}
    if include_truth:
        required.add("TVT")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{well} horizontal missing columns {missing}")
    return frame


def load_typewell(data_dir: Path, well: str) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"{well}__typewell.csv")
    if not {"TVT", "GR"}.issubset(frame.columns):
        raise ValueError(f"{well} typewell must contain TVT and GR")
    frame = frame[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["TVT"]).sort_values("TVT").reset_index(drop=True)


def validate_frozen_contract(config: dict[str, Any]) -> None:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "method_fidelity.classification": "proxy",
        "model.active_variants": list(FAMILY_ORDER),
        "model.emission.degrees_of_freedom": 1.0,
        "model.prefix.initial_rate_rows": 256,
        "model.prefix.in_model_rows": 128,
        "model.reference.coarse_sibling_bin_ft": 0.25,
        "model.reference.fine_sibling_bin_ft": 0.0625,
        "execution_contract.scientific_variants": 3,
        "execution_contract.maximum_train_hmm_well_runs": 2319,
        "execution_contract.lightgbm_configs": 0,
        "execution_contract.boosters": 0,
        "execution_contract.pf_runs": 0,
        "execution_contract.beam_runs": 0,
        "late_submission.enabled": True,
        "late_submission.phase": "post_competition_late_submission",
        "late_submission.allow_lb_retuning": False,
    }
    for key, value in expected.items():
        actual = get_nested(config, key)
        if actual != value:
            raise ValueError(f"frozen contract mismatch {key}: expected={value!r} actual={actual!r}")
    weights = get_nested(config, "model.family_weights") or {}
    if tuple(weights) != FAMILY_ORDER:
        raise ValueError(f"family weight order must be {FAMILY_ORDER}")
    if not math.isclose(sum(float(weights[name]) for name in FAMILY_ORDER), 1.0, abs_tol=1e-12):
        raise ValueError("family weights must sum to one")


# %% [markdown]
# ## 3. Fold-safe sibling-reference construction

# %%
def load_group_map(config: dict[str, Any], assignment_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    frame = pd.read_csv(assignment_path, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"group assignment missing {missing}")
    method = str(get_nested(config, "data.typewell_group_assignment.method"))
    threshold = str(get_nested(config, "data.typewell_group_assignment.threshold"))
    min_size = int(get_nested(config, "data.typewell_group_assignment.minimum_cluster_size"))
    size = pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0).astype(int)
    selected = frame[
        frame["method"].astype(str).eq(method)
        & frame["threshold"].astype(str).eq(threshold)
        & size.ge(min_size)
    ].copy()
    selected = selected.sort_values(["cluster_id", "well_id"])
    if selected["well_id"].duplicated().any():
        raise ValueError("selected group assignment contains duplicate well ids")
    mapping = dict(zip(selected["well_id"].astype(str), selected["cluster_id"].astype(str), strict=False))
    meta = {
        "path": str(assignment_path),
        "sha256": sha256_path(assignment_path),
        "method": method,
        "threshold": threshold,
        "minimum_cluster_size": min_size,
        "wells": len(mapping),
        "groups": int(selected["cluster_id"].nunique()),
        "max_group_size": int(selected.groupby("cluster_id").size().max()),
    }
    return mapping, meta


def make_fold_assignment(wells: list[str], n_folds: int) -> dict[str, int]:
    ordered = np.asarray(sorted(wells), dtype=object)
    dummy = np.zeros(len(ordered), dtype=np.float32)
    assignment: dict[str, int] = {}
    splitter = GroupKFold(n_splits=n_folds)
    for fold, (_, valid_index) in enumerate(splitter.split(dummy, dummy, groups=ordered)):
        for index in valid_index:
            assignment[str(ordered[index])] = int(fold)
    if set(assignment) != set(wells):
        raise RuntimeError("fold assignment did not cover every well")
    return assignment


def collect_sibling_rows(
    train_dir: Path,
    wells: list[str],
    fold_map: dict[str, int],
    group_map: dict[str, str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    started = time.time()
    for index, well in enumerate(sorted(wells), start=1):
        group = group_map.get(well)
        if group is None:
            continue
        frame = pd.read_csv(
            train_dir / f"{well}__horizontal_well.csv",
            usecols=["TVT", "GR"],
        ).apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["TVT", "GR"])
        if frame.empty:
            continue
        pieces.append(
            pd.DataFrame(
                {
                    "group": group,
                    "source_fold": np.int8(fold_map[well]),
                    "tvt": frame["TVT"].to_numpy(np.float32),
                    "gr": frame["GR"].to_numpy(np.float32),
                }
            )
        )
        if index % 100 == 0:
            print(f"[sibling rows] {index}/{len(wells)} elapsed={time.time() - started:.1f}s", flush=True)
    if not pieces:
        raise RuntimeError("no sibling horizontal rows were collected")
    output = pd.concat(pieces, ignore_index=True)
    if not np.isfinite(output[["tvt", "gr"]].to_numpy(np.float64)).all():
        raise RuntimeError("sibling source contains non-finite TVT/GR")
    return output


def aggregate_reference_rows(rows: pd.DataFrame, bin_width: float) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    work = rows.copy()
    work["bin"] = np.floor(work["tvt"].to_numpy(np.float64) / bin_width).astype(np.int64)
    grouped = (
        work.groupby(["group", "bin"], sort=True, observed=True)["gr"]
        .agg(["median", "count"])
        .reset_index()
    )
    references: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group, frame in grouped.groupby("group", sort=True, observed=True):
        bins = frame["bin"].to_numpy(np.int64)
        references[str(group)] = (
            ((bins.astype(np.float64) + 0.5) * bin_width).astype(np.float64),
            frame["median"].to_numpy(np.float64),
            frame["count"].to_numpy(np.int32),
        )
    return references


def build_fold_atlases(
    sibling_rows: pd.DataFrame,
    folds: list[int],
    coarse_width: float,
    fine_width: float,
) -> dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    output: dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}
    for fold in sorted(set(folds)):
        source = sibling_rows[sibling_rows["source_fold"].ne(fold)]
        if source.empty or source["source_fold"].eq(fold).any():
            raise RuntimeError(f"fold {fold} sibling exclusion failed")
        started = time.time()
        output[int(fold)] = {
            "coarse": aggregate_reference_rows(source, coarse_width),
            "fine": aggregate_reference_rows(source, fine_width),
        }
        print(
            f"[atlas] fold={fold} rows={len(source)} coarse_groups={len(output[fold]['coarse'])} "
            f"fine_groups={len(output[fold]['fine'])} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    return output


def interpolate_reference(
    states: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if reference is None or len(reference[0]) < 2:
        return np.full(states.shape, np.nan, dtype=np.float64)
    x, y, _ = reference
    return np.interp(states.ravel(), x, y, left=np.nan, right=np.nan).reshape(states.shape)


def binned_prefix_reference(horizontal: pd.DataFrame, bin_width: float) -> tuple[np.ndarray, np.ndarray] | None:
    known = horizontal[horizontal["TVT_input"].notna()].copy()
    tvt = pd.to_numeric(known["TVT_input"], errors="coerce")
    gr = pd.to_numeric(known["GR"], errors="coerce")
    valid = tvt.notna() & gr.notna()
    if int(valid.sum()) < 2:
        return None
    work = pd.DataFrame({"tvt": tvt[valid], "gr": gr[valid]})
    work["bin"] = np.floor(work["tvt"].to_numpy(np.float64) / bin_width).astype(np.int64)
    grouped = work.groupby("bin", sort=True)["gr"].median()
    x = (grouped.index.to_numpy(np.float64) + 0.5) * bin_width
    y = grouped.to_numpy(np.float64)
    return (x, y) if len(x) >= 2 else None


def reference_at(states: np.ndarray, curve: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if curve is None or len(curve[0]) < 2:
        return np.full(states.shape, np.nan, dtype=np.float64)
    return np.interp(states.ravel(), curve[0], curve[1], left=np.nan, right=np.nan).reshape(states.shape)


# %% [markdown]
# ## 4. Joint TVT/rate/bias/reference exact forward-backward kernel

# %%
def _nearest_transition_logs(
    dm: np.ndarray,
    values: np.ndarray,
    sigma_per_sqrt_ft: float,
    momentum: float,
) -> np.ndarray:
    count = len(values)
    step = float(values[1] - values[0]) if count > 1 else 1.0
    logs = np.full((len(dm), count, 3), -1e18, dtype=np.float64)
    for t, delta_md in enumerate(dm):
        variance_cells = (sigma_per_sqrt_ft * math.sqrt(max(float(delta_md), 1e-12)) / step) ** 2
        for source in range(count):
            mean_move = -(1.0 - momentum) * values[source] * float(delta_md) / step
            p_plus = max(0.5 * (variance_cells + mean_move), 1e-12) if source + 1 < count else 0.0
            p_minus = max(0.5 * (variance_cells - mean_move), 1e-12) if source > 0 else 0.0
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            p_stay = 1.0 - p_plus - p_minus
            if source > 0:
                logs[t, source, 0] = math.log(max(p_minus, 1e-30))
            logs[t, source, 1] = math.log(max(p_stay, 1e-30))
            if source + 1 < count:
                logs[t, source, 2] = math.log(max(p_plus, 1e-30))
    return logs


def prepare_transition_logs(
    dm: np.ndarray,
    rate_delta: np.ndarray,
    bias_delta: np.ndarray,
    config: dict[str, Any],
    reference_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transition = get_nested(config, "model.transition") or {}
    rate_log = _nearest_transition_logs(
        dm,
        rate_delta,
        float(transition["rate_sigma_per_sqrt_ft"]),
        float(transition["rate_momentum"]),
    )
    bias_log = _nearest_transition_logs(
        dm,
        bias_delta,
        float(transition["bias_sigma_gr_per_sqrt_ft"]),
        1.0,
    )
    offset_step = float(get_nested(config, "model.state.offset_step_ft"))
    sigma_position = max(float(transition["position_sigma_ft"]), 0.35 * offset_step)
    position_b0 = np.empty((len(dm), len(rate_delta)), dtype=np.int32)
    position_log = np.empty((len(dm), len(rate_delta), 5), dtype=np.float64)
    for t, delta_md in enumerate(dm):
        for rate_index, rate in enumerate(rate_delta):
            mu = float(rate) * float(delta_md)
            center = int(math.floor(mu / offset_step + 0.5))
            position_b0[t, rate_index] = center
            values = np.empty(5, dtype=np.float64)
            for kernel_index in range(5):
                delta = (center - 2 + kernel_index) * offset_step - mu
                values[kernel_index] = -0.5 * (delta / sigma_position) ** 2
            maximum = float(values.max())
            values -= maximum + math.log(float(np.exp(values - maximum).sum()))
            position_log[t, rate_index] = values
    if reference_count == 1:
        reference_log = np.zeros((1, 1), dtype=np.float64)
    else:
        switch = float(transition["reference_switch_probability"])
        reference_log = np.full(
            (reference_count, reference_count),
            math.log(max(switch / (reference_count - 1), 1e-30)),
            dtype=np.float64,
        )
        np.fill_diagonal(reference_log, math.log(max(1.0 - switch, 1e-30)))
    return rate_log, bias_log, position_b0, position_log, reference_log


@njit(cache=True, nogil=True, parallel=True)
def joint_hmm_forward_backward(
    emission,
    offsets,
    rate_log,
    bias_log,
    position_b0,
    position_log,
    reference_log,
    position_prior,
    rate_prior,
    bias_prior,
):
    """Exact sum-product on (TVT offset, rate, GR bias, reference family)."""
    t_count, p_count, b_count, f_count = emission.shape
    r_count = rate_log.shape[1]
    neg = np.float32(-1e30)
    alpha = np.full((t_count, p_count, r_count, b_count, f_count), neg, np.float32)
    prev = np.empty((p_count, r_count, b_count, f_count), np.float32)
    for p in prange(p_count):
        for r in range(r_count):
            for b in range(b_count):
                for f in range(f_count):
                    prev[p, r, b, f] = np.float32(
                        position_prior[p] + rate_prior[r] + bias_prior[b] - math.log(f_count)
                    )

    tmp_r = np.empty_like(prev)
    tmp_b = np.empty_like(prev)
    tmp_f = np.empty_like(prev)
    cur = np.empty_like(prev)
    scales = np.empty(t_count, np.float64)

    for t in range(t_count):
        for p in prange(p_count):
            for r2 in range(r_count):
                r0 = max(0, r2 - 1)
                r1 = min(r_count - 1, r2 + 1)
                for b in range(b_count):
                    for f in range(f_count):
                        best = neg
                        for rs in range(r0, r1 + 1):
                            value = prev[p, rs, b, f] + rate_log[t, rs, r2 - rs + 1]
                            if value > best:
                                best = value
                        total = 0.0
                        for rs in range(r0, r1 + 1):
                            total += math.exp(prev[p, rs, b, f] + rate_log[t, rs, r2 - rs + 1] - best)
                        tmp_r[p, r2, b, f] = np.float32(best + math.log(total))

        for p in prange(p_count):
            for r in range(r_count):
                for b2 in range(b_count):
                    b0 = max(0, b2 - 1)
                    b1 = min(b_count - 1, b2 + 1)
                    for f in range(f_count):
                        best = neg
                        for bs in range(b0, b1 + 1):
                            value = tmp_r[p, r, bs, f] + bias_log[t, bs, b2 - bs + 1]
                            if value > best:
                                best = value
                        total = 0.0
                        for bs in range(b0, b1 + 1):
                            total += math.exp(tmp_r[p, r, bs, f] + bias_log[t, bs, b2 - bs + 1] - best)
                        tmp_b[p, r, b2, f] = np.float32(best + math.log(total))

        for p in prange(p_count):
            for r in range(r_count):
                for b in range(b_count):
                    for f2 in range(f_count):
                        best = neg
                        for fs in range(f_count):
                            value = tmp_b[p, r, b, fs] + reference_log[fs, f2]
                            if value > best:
                                best = value
                        total = 0.0
                        for fs in range(f_count):
                            total += math.exp(tmp_b[p, r, b, fs] + reference_log[fs, f2] - best)
                        tmp_f[p, r, b, f2] = np.float32(best + math.log(total))

        for p2 in prange(p_count):
            for r in range(r_count):
                center = position_b0[t, r]
                for b in range(b_count):
                    for f in range(f_count):
                        best = neg
                        for k in range(5):
                            p1 = p2 - (center - 2 + k)
                            if 0 <= p1 < p_count:
                                value = tmp_f[p1, r, b, f] + position_log[t, r, k]
                                if value > best:
                                    best = value
                        total = 0.0
                        for k in range(5):
                            p1 = p2 - (center - 2 + k)
                            if 0 <= p1 < p_count:
                                total += math.exp(tmp_f[p1, r, b, f] + position_log[t, r, k] - best)
                        cur[p2, r, b, f] = np.float32(best + math.log(total) + emission[t, p2, b, f])

        scale = -1e30
        for p in range(p_count):
            for r in range(r_count):
                for b in range(b_count):
                    for f in range(f_count):
                        if cur[p, r, b, f] > scale:
                            scale = cur[p, r, b, f]
        scales[t] = scale
        for p in prange(p_count):
            for r in range(r_count):
                for b in range(b_count):
                    for f in range(f_count):
                        value = np.float32(cur[p, r, b, f] - scale)
                        alpha[t, p, r, b, f] = value
                        prev[p, r, b, f] = value

    mean_offset = np.empty(t_count, np.float64)
    std_offset = np.empty(t_count, np.float64)
    boundary_mass = np.empty(t_count, np.float64)
    normalization_error = np.empty(t_count, np.float64)
    beta_next = np.zeros((p_count, r_count, b_count, f_count), np.float32)
    beta_pos = np.empty_like(beta_next)
    beta_ref = np.empty_like(beta_next)
    beta_bias = np.empty_like(beta_next)
    beta_cur = np.empty_like(beta_next)

    for t in range(t_count - 1, -1, -1):
        best = -1e30
        for p in range(p_count):
            for r in range(r_count):
                for b in range(b_count):
                    for f in range(f_count):
                        value = alpha[t, p, r, b, f] + beta_next[p, r, b, f]
                        if value > best:
                            best = value
        total = 0.0
        weighted = 0.0
        weighted2 = 0.0
        edge = 0.0
        for p in range(p_count):
            p_total = 0.0
            for r in range(r_count):
                for b in range(b_count):
                    for f in range(f_count):
                        p_total += math.exp(alpha[t, p, r, b, f] + beta_next[p, r, b, f] - best)
            total += p_total
            weighted += p_total * offsets[p]
            weighted2 += p_total * offsets[p] * offsets[p]
            if p == 0 or p == p_count - 1:
                edge += p_total
        mean = weighted / total
        mean_offset[t] = mean
        std_offset[t] = math.sqrt(max(weighted2 / total - mean * mean, 0.0))
        boundary_mass[t] = edge / total
        normalization_error[t] = abs(total / total - 1.0)
        if t == 0:
            break

        for p1 in prange(p_count):
            for r2 in range(r_count):
                center = position_b0[t, r2]
                for b2 in range(b_count):
                    for f2 in range(f_count):
                        best2 = neg
                        for k in range(5):
                            p2 = p1 + (center - 2 + k)
                            if 0 <= p2 < p_count:
                                value = position_log[t, r2, k] + emission[t, p2, b2, f2] + beta_next[p2, r2, b2, f2]
                                if value > best2:
                                    best2 = value
                        total2 = 0.0
                        for k in range(5):
                            p2 = p1 + (center - 2 + k)
                            if 0 <= p2 < p_count:
                                total2 += math.exp(
                                    position_log[t, r2, k]
                                    + emission[t, p2, b2, f2]
                                    + beta_next[p2, r2, b2, f2]
                                    - best2
                                )
                        beta_pos[p1, r2, b2, f2] = np.float32(best2 + math.log(total2))

        for p in prange(p_count):
            for r in range(r_count):
                for b in range(b_count):
                    for f1 in range(f_count):
                        best2 = neg
                        for f2 in range(f_count):
                            value = reference_log[f1, f2] + beta_pos[p, r, b, f2]
                            if value > best2:
                                best2 = value
                        total2 = 0.0
                        for f2 in range(f_count):
                            total2 += math.exp(reference_log[f1, f2] + beta_pos[p, r, b, f2] - best2)
                        beta_ref[p, r, b, f1] = np.float32(best2 + math.log(total2))

        for p in prange(p_count):
            for r in range(r_count):
                for b1 in range(b_count):
                    b0 = max(0, b1 - 1)
                    b1max = min(b_count - 1, b1 + 1)
                    for f in range(f_count):
                        best2 = neg
                        for b2 in range(b0, b1max + 1):
                            value = bias_log[t, b1, b2 - b1 + 1] + beta_ref[p, r, b2, f]
                            if value > best2:
                                best2 = value
                        total2 = 0.0
                        for b2 in range(b0, b1max + 1):
                            total2 += math.exp(bias_log[t, b1, b2 - b1 + 1] + beta_ref[p, r, b2, f] - best2)
                        beta_bias[p, r, b1, f] = np.float32(best2 + math.log(total2))

        for p in prange(p_count):
            for r1 in range(r_count):
                r0 = max(0, r1 - 1)
                r1max = min(r_count - 1, r1 + 1)
                for b in range(b_count):
                    for f in range(f_count):
                        best2 = neg
                        for r2 in range(r0, r1max + 1):
                            value = rate_log[t, r1, r2 - r1 + 1] + beta_bias[p, r2, b, f]
                            if value > best2:
                                best2 = value
                        total2 = 0.0
                        for r2 in range(r0, r1max + 1):
                            total2 += math.exp(rate_log[t, r1, r2 - r1 + 1] + beta_bias[p, r2, b, f] - best2)
                        beta_cur[p, r1, b, f] = np.float32(best2 + math.log(total2) - scales[t])
        for p in prange(p_count):
            for r in range(r_count):
                for b in range(b_count):
                    for f in range(f_count):
                        beta_next[p, r, b, f] = beta_cur[p, r, b, f]

    last_best = -1e30
    for p in range(p_count):
        for r in range(r_count):
            for b in range(b_count):
                for f in range(f_count):
                    if alpha[t_count - 1, p, r, b, f] > last_best:
                        last_best = alpha[t_count - 1, p, r, b, f]
    last_total = 0.0
    for p in range(p_count):
        for r in range(r_count):
            for b in range(b_count):
                for f in range(f_count):
                    last_total += math.exp(alpha[t_count - 1, p, r, b, f] - last_best)
    loglik = float(float(np.sum(scales)) + float(last_best) + math.log(last_total))
    return mean_offset, std_offset, boundary_mass, normalization_error, loglik


# %% [markdown]
# ## 5. Three HMM families and physical projection

# %%
def robust_initial_rate(horizontal: pd.DataFrame, rows: int) -> tuple[float, np.ndarray]:
    known = horizontal[horizontal["TVT_input"].notna()].tail(rows)
    md = pd.to_numeric(known["MD"], errors="coerce").to_numpy(np.float64)
    u = (
        pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(np.float64)
        + pd.to_numeric(known["Z"], errors="coerce").to_numpy(np.float64)
    )
    dmd = np.diff(md)
    rates = np.diff(u) / np.where(dmd > 0, dmd, np.nan)
    rates = rates[np.isfinite(rates) & (dmd > 0)]
    rates = rates[np.abs(rates) <= 0.25]
    if len(rates) < 3:
        return 0.0, np.asarray([0.0], dtype=np.float64)
    low, high = np.quantile(rates, [0.01, 0.99])
    trimmed = rates[(rates >= low) & (rates <= high)]
    return float(np.median(trimmed if len(trimmed) else rates)), rates


def sequence_contract(horizontal: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    prefix_rows = int(get_nested(config, "model.prefix.in_model_rows"))
    rate_rows = int(get_nested(config, "model.prefix.initial_rate_rows"))
    known_index = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())
    eval_index = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    if len(known_index) == 0 or len(eval_index) == 0:
        raise ValueError("well needs both known prefix and unknown suffix")
    prefix_index = known_index[-min(prefix_rows, len(known_index)) :]
    sequence_index = np.concatenate([prefix_index, eval_index]).astype(np.int64)
    initial_rate, prefix_rates = robust_initial_rate(horizontal, rate_rows)
    first = int(prefix_index[0])
    md = pd.to_numeric(horizontal.loc[sequence_index, "MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(horizontal.loc[sequence_index, "Z"], errors="coerce").to_numpy(np.float64)
    gr = pd.to_numeric(horizontal.loc[sequence_index, "GR"], errors="coerce").to_numpy(np.float64)
    first_u = float(horizontal.loc[first, "TVT_input"] + horizontal.loc[first, "Z"])
    baseline_u = first_u + initial_rate * (md - md[0])
    baseline_tvt = baseline_u - z
    dm = np.diff(md, prepend=md[0])
    if len(dm) > 1:
        dm[0] = max(float(np.median(dm[1:][dm[1:] > 0])), 1.0)
    else:
        dm[0] = 1.0
    if np.any(dm <= 0) or not np.isfinite(dm).all():
        raise ValueError("MD must be finite and strictly increasing inside the HMM sequence")
    return {
        "known_index": known_index,
        "eval_index": eval_index,
        "prefix_index": prefix_index,
        "sequence_index": sequence_index,
        "prefix_count": len(prefix_index),
        "initial_rate": initial_rate,
        "prefix_rates": prefix_rates,
        "md": md,
        "z": z,
        "gr": gr,
        "baseline_tvt": baseline_tvt,
        "dm": dm,
    }


def robust_sigma_and_bias(horizontal: pd.DataFrame, typewell: pd.DataFrame, config: dict[str, Any]) -> tuple[float, float]:
    known = horizontal[horizontal["TVT_input"].notna()]
    known_tvt = pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").to_numpy(np.float64)
    type_tvt = typewell["TVT"].to_numpy(np.float64)
    type_gr = typewell["GR"].interpolate(limit_direction="both").fillna(0.0).to_numpy(np.float64)
    reference = np.interp(known_tvt, type_tvt, type_gr)
    residual = known_gr - reference
    residual = residual[np.isfinite(residual)]
    center = float(np.median(residual)) if len(residual) else 0.0
    mad = float(1.4826 * np.median(np.abs(residual - center))) if len(residual) else 30.0
    low, high = [float(value) for value in get_nested(config, "model.emission.sigma_clip_gr")]
    return float(np.clip(mad, low, high)), center


def mix_self_reference(base: np.ndarray, self_values: np.ndarray, weight: float) -> np.ndarray:
    output = base.copy()
    valid = np.isfinite(self_values)
    output[valid] = (1.0 - weight) * output[valid] + weight * self_values[valid]
    return output


def build_reference_tensor(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    sequence: dict[str, Any],
    offsets: np.ndarray,
    group: str | None,
    fold_atlas: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    family: str,
    config: dict[str, Any],
) -> np.ndarray:
    states = sequence["baseline_tvt"][:, None] + offsets[None, :]
    type_tvt = typewell["TVT"].to_numpy(np.float64)
    type_gr = typewell["GR"].interpolate(limit_direction="both").fillna(0.0).to_numpy(np.float64)
    prefix_curve = binned_prefix_reference(
        horizontal,
        float(get_nested(config, "model.reference.coarse_sibling_bin_ft")),
    )
    support = int(horizontal["TVT_input"].notna().sum())
    shrink = float(get_nested(config, "model.reference.self_reference_shrinkage_rows"))
    max_weight = float(get_nested(config, "model.reference.self_reference_max_weight"))
    self_weight = min(max_weight, support / max(support + shrink, 1.0))

    def type_values(query: np.ndarray) -> np.ndarray:
        return np.interp(query.ravel(), type_tvt, type_gr).reshape(query.shape)

    def sibling_values(query: np.ndarray, scale: str) -> np.ndarray:
        reference = fold_atlas.get(scale, {}).get(str(group)) if group is not None else None
        values = interpolate_reference(query, reference)
        fallback = type_values(query)
        return np.where(np.isfinite(values), values, fallback)

    def with_self(query: np.ndarray, base: np.ndarray) -> np.ndarray:
        return mix_self_reference(base, reference_at(query, prefix_curve), self_weight)

    if family == "exp417_base_reconstruction":
        sibling = sibling_values(states, "coarse")
        type_ref = type_values(states)
        references = []
        mixtures = get_nested(config, "model.reference.base_typewell_sibling_mixtures")
        for type_weight, sibling_weight in mixtures:
            references.append(
                with_self(states, float(type_weight) * type_ref + float(sibling_weight) * sibling)
            )
        return np.stack(references, axis=-1).astype(np.float64)
    if family == "local_dtw_reconstruction":
        anchor = float(horizontal.loc[sequence["known_index"][-1], "TVT_input"])
        references = []
        for stretch in get_nested(config, "model.reference.local_dtw_stretches"):
            query = anchor + float(stretch) * (states - anchor)
            sibling = sibling_values(query, "coarse")
            base = 0.2 * type_values(query) + 0.8 * sibling
            references.append(with_self(query, base))
        return np.stack(references, axis=-1).astype(np.float64)
    if family == "fine_bin_reconstruction":
        sibling = sibling_values(states, "fine")
        return with_self(states, sibling)[..., None].astype(np.float64)
    raise ValueError(f"unknown HMM family {family}")


def build_emission(
    horizontal: pd.DataFrame,
    sequence: dict[str, Any],
    references: np.ndarray,
    biases: np.ndarray,
    sigma_gr: float,
    config: dict[str, Any],
) -> np.ndarray:
    gr = sequence["gr"]
    residual = gr[:, None, None, None] - (
        references[:, :, None, :] + biases[None, None, :, None]
    )
    emission = -np.log1p((residual / sigma_gr) ** 2)
    missing = ~np.isfinite(gr)
    emission[missing] = float(get_nested(config, "model.emission.missing_gr_log_likelihood"))
    prefix_count = int(sequence["prefix_count"])
    clamp_sigma = float(get_nested(config, "model.prefix.tvt_clamp_sigma_ft"))
    prefix_index = sequence["prefix_index"]
    observed = pd.to_numeric(horizontal.loc[prefix_index, "TVT_input"], errors="coerce").to_numpy(np.float64)
    state_tvt = sequence["baseline_tvt"][:prefix_count, None]
    offsets = np.linspace(
        float(get_nested(config, "model.state.offset_min_ft")),
        float(get_nested(config, "model.state.offset_max_ft")),
        int(round((float(get_nested(config, "model.state.offset_max_ft")) - float(get_nested(config, "model.state.offset_min_ft"))) / float(get_nested(config, "model.state.offset_step_ft")))) + 1,
    )
    clamp = np.maximum(-0.5 * ((state_tvt + offsets[None, :] - observed[:, None]) / clamp_sigma) ** 2, -60.0)
    emission[:prefix_count] += clamp[:, :, None, None]
    return emission.astype(np.float32)


def physical_projection(
    raw_prediction: np.ndarray,
    horizontal: pd.DataFrame,
    eval_index: np.ndarray,
    prefix_rates: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    last_known = int(np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())[-1])
    last_tvt = float(horizontal.loc[last_known, "TVT_input"])
    last_z = float(horizontal.loc[last_known, "Z"])
    last_md = float(horizontal.loc[last_known, "MD"])
    md = pd.to_numeric(horizontal.loc[eval_index, "MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(horizontal.loc[eval_index, "Z"], errors="coerce").to_numpy(np.float64)
    full_md = np.concatenate([[last_md], md])
    full_u = np.concatenate([[last_tvt + last_z], raw_prediction + z])
    dmd = np.diff(full_md)
    raw_rate = np.diff(full_u) / dmd
    q_low, q_high = [float(value) for value in get_nested(config, "model.projection.prefix_rate_quantiles")]
    low, high = np.quantile(prefix_rates, [q_low, q_high]) if len(prefix_rates) else (-0.25, 0.25)
    absolute_cap = float(get_nested(config, "model.projection.absolute_formation_rate_cap"))
    low = max(float(low), -absolute_cap)
    high = min(float(high), absolute_cap)
    if low > high:
        low, high = -absolute_cap, absolute_cap
    clipped_rate = np.clip(raw_rate, low, high)
    projected_u = (last_tvt + last_z) + np.cumsum(clipped_rate * dmd)
    projected_tvt = projected_u - z
    correction_cap = float(get_nested(config, "model.projection.integrated_tvt_correction_cap_ft"))
    correction = np.clip(projected_tvt - raw_prediction, -correction_cap, correction_cap)
    output = raw_prediction + correction
    return output, {
        "prefix_rate_low": low,
        "prefix_rate_high": high,
        "raw_rate_clip_fraction": float(np.mean(raw_rate != clipped_rate)),
        "integrated_correction_abs_max": float(np.max(np.abs(correction))),
    }


def decode_family(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    sequence: dict[str, Any],
    group: str | None,
    fold_atlas: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    family: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    state = get_nested(config, "model.state") or {}
    offsets = np.arange(
        float(state["offset_min_ft"]),
        float(state["offset_max_ft"]) + 0.5 * float(state["offset_step_ft"]),
        float(state["offset_step_ft"]),
        dtype=np.float64,
    )
    rate_delta = np.linspace(
        float(state["rate_delta_min"]),
        float(state["rate_delta_max"]),
        int(state["rate_states"]),
        dtype=np.float64,
    )
    bias_delta = np.linspace(
        float(state["bias_delta_min_gr"]),
        float(state["bias_delta_max_gr"]),
        int(state["bias_states"]),
        dtype=np.float64,
    )
    sigma_gr, bias_center = robust_sigma_and_bias(horizontal, typewell, config)
    biases = bias_center + bias_delta
    references = build_reference_tensor(
        horizontal,
        typewell,
        sequence,
        offsets,
        group,
        fold_atlas,
        family,
        config,
    )
    emission = build_emission(horizontal, sequence, references, biases, sigma_gr, config)
    rate_log, bias_log, position_b0, position_log, reference_log = prepare_transition_logs(
        sequence["dm"], rate_delta, bias_delta, config, references.shape[-1]
    )
    prefix = get_nested(config, "model.prefix") or {}
    position_prior = -0.5 * (offsets / float(prefix["initial_position_sigma_ft"])) ** 2
    rate_prior = -0.5 * (rate_delta / float(prefix["initial_rate_sigma"])) ** 2
    bias_prior = -0.5 * (bias_delta / float(prefix["initial_bias_sigma_gr"])) ** 2
    started = time.time()
    mean_offset, std_offset, boundary_mass, norm_error, loglik = joint_hmm_forward_backward(
        emission,
        offsets,
        rate_log,
        bias_log,
        position_b0,
        position_log,
        reference_log,
        position_prior,
        rate_prior,
        bias_prior,
    )
    prefix_count = int(sequence["prefix_count"])
    raw = sequence["baseline_tvt"][prefix_count:] + mean_offset[prefix_count:]
    projected, projection_meta = physical_projection(
        raw,
        horizontal,
        sequence["eval_index"],
        sequence["prefix_rates"],
        config,
    )
    return {
        "prediction": projected,
        "raw_prediction": raw,
        "posterior_std": std_offset[prefix_count:],
        "boundary_mass": boundary_mass[prefix_count:],
        "normalization_error": norm_error[prefix_count:],
        "loglik": float(loglik),
        "sigma_gr": sigma_gr,
        "bias_center": bias_center,
        "reference_states": references.shape[-1],
        "state_count": len(offsets) * len(rate_delta) * len(biases) * references.shape[-1],
        "elapsed_seconds": time.time() - started,
        **projection_meta,
    }


def decode_well_target_free(
    safe_horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    group: str | None,
    fold_atlas: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "TVT" in safe_horizontal.columns:
        raise ValueError("target-free decoder received forbidden TVT column")
    sequence = sequence_contract(safe_horizontal, config)
    weights = get_nested(config, "model.family_weights")
    family_results: dict[str, dict[str, Any]] = {}
    for family in FAMILY_ORDER:
        family_results[family] = decode_family(
            safe_horizontal,
            typewell,
            sequence,
            group,
            fold_atlas,
            family,
            config,
        )
    final = np.zeros(len(sequence["eval_index"]), dtype=np.float64)
    for family in FAMILY_ORDER:
        final += float(weights[family]) * family_results[family]["prediction"]
    last_known = int(sequence["known_index"][-1])
    last_tvt = float(safe_horizontal.loc[last_known, "TVT_input"])
    frame: dict[str, Any] = {
        "row_idx": sequence["eval_index"].astype(np.int64),
        "prediction": final,
        "last_known_tvt": last_tvt,
    }
    for family in FAMILY_ORDER:
        frame[f"{family}_prediction"] = family_results[family]["prediction"]
        frame[f"{family}_posterior_std"] = family_results[family]["posterior_std"]
        frame[f"{family}_boundary_mass"] = family_results[family]["boundary_mass"]
    output = pd.DataFrame(frame)
    meta = {
        "rows": len(output),
        "prefix_rows": int(sequence["prefix_count"]),
        "initial_rate": float(sequence["initial_rate"]),
        "group": group,
        "family": {
            family: {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
            for family, result in family_results.items()
        },
        "normalization_error_max": max(
            float(np.max(result["normalization_error"])) for result in family_results.values()
        ),
        "boundary_mass_max": max(
            float(np.max(result["boundary_mass"])) for result in family_results.values()
        ),
    }
    return output, meta


# %% [markdown]
# ## 6. Fold-safe OOF orchestration and prediction freeze

# %%
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def run_oof(config: dict[str, Any]) -> dict[str, Any]:
    validate_frozen_contract(config)
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for the joint exact HMM")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    competition_root = resolve_competition_root(config)
    train_dir = competition_root / "train"
    wells = list_wells(train_dir)
    expected_wells = int(get_nested(config, "execution_contract.expected_train_wells"))
    if len(wells) != expected_wells:
        raise RuntimeError(f"expected {expected_wells} train wells, found {len(wells)}")
    fold_map = make_fold_assignment(wells, int(get_nested(config, "validation.n_folds")))
    assignment_path = resolve_group_assignment(config)
    group_map, group_meta = load_group_map(config, assignment_path)

    debug_max = int(os.environ.get("EXP515_DEBUG_MAX_WELLS", "0"))
    target_wells = sorted(wells)[:debug_max] if debug_max > 0 else sorted(wells)
    target_folds = sorted({fold_map[well] for well in target_wells})
    sibling_rows = collect_sibling_rows(train_dir, wells, fold_map, group_map)
    atlas = build_fold_atlases(
        sibling_rows,
        target_folds,
        float(get_nested(config, "model.reference.coarse_sibling_bin_ft")),
        float(get_nested(config, "model.reference.fine_sibling_bin_ft")),
    )
    sibling_row_count = len(sibling_rows)
    del sibling_rows

    predictions: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    started = time.time()
    truth_rows_accessed_before_prediction_freeze = 0
    for index, well in enumerate(target_wells, start=1):
        full_horizontal = load_horizontal(train_dir, well, include_truth=True)
        truth = pd.to_numeric(full_horizontal["TVT"], errors="coerce").to_numpy(np.float64)
        safe_horizontal = full_horizontal.drop(columns=["TVT"])
        typewell = load_typewell(train_dir, well)
        fold = fold_map[well]
        decoded, meta = decode_well_target_free(
            safe_horizontal,
            typewell,
            group_map.get(well),
            atlas[fold],
            config,
        )
        decoded.insert(0, "well_id", well)
        decoded.insert(1, "fold", fold)
        decoded.insert(2, "id", [f"{well}_{row}" for row in decoded["row_idx"]])
        decoded = decoded.sort_values("row_idx").reset_index(drop=True)
        prediction_sha = dataframe_content_sha(
            decoded,
            ["well_id", "fold", "id", "row_idx", "prediction", *[f"{family}_prediction" for family in FAMILY_ORDER]],
        )
        decoded["tvt_true"] = truth[decoded["row_idx"].to_numpy(np.int64)]
        decoded["prediction_freeze_sha"] = prediction_sha
        predictions.append(decoded)
        runtime_rows.append(
            {
                "well_id": well,
                "fold": fold,
                "status": "ok",
                "prediction_freeze_sha": prediction_sha,
                **meta,
            }
        )
        print(
            f"[OOF {index}/{len(target_wells)}] well={well} fold={fold} rows={len(decoded)} "
            f"rmse={rmse(decoded['tvt_true'], decoded['prediction']):.6f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    oof = pd.concat(predictions, ignore_index=True).sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    runtime = pd.DataFrame(runtime_rows).sort_values("well_id").reset_index(drop=True)
    if oof["id"].duplicated().any() or not np.isfinite(oof["prediction"]).all():
        raise RuntimeError("OOF ID/finite contract failed")
    if truth_rows_accessed_before_prediction_freeze != 0:
        raise RuntimeError("truth was accessed before prediction freeze")
    return {
        "oof": oof,
        "runtime": runtime,
        "group_meta": group_meta,
        "fold_map": fold_map,
        "target_wells": target_wells,
        "sibling_row_count": sibling_row_count,
        "competition_root": competition_root,
        "truth_rows_accessed_before_prediction_freeze": truth_rows_accessed_before_prediction_freeze,
        "elapsed_seconds": time.time() - started,
        "debug": debug_max > 0,
    }


# %% [markdown]
# ## 7. Metrics, diagnostics, and generated artifacts

# %%
def summarize_and_write(run: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    oof = run["oof"]
    runtime = run["runtime"]
    artifacts = artifact_dir()
    pooled = rmse(oof["tvt_true"], oof["prediction"])
    family_rmse = {
        family: rmse(oof["tvt_true"], oof[f"{family}_prediction"])
        for family in FAMILY_ORDER
    }
    fold_rows = [
        {
            "fold": int(fold),
            "rows": int(len(frame)),
            "wells": int(frame["well_id"].nunique()),
            "rmse": rmse(frame["tvt_true"], frame["prediction"]),
        }
        for fold, frame in oof.groupby("fold", sort=True)
    ]
    by_well = (
        oof.groupby(["well_id", "fold"], sort=True)
        .apply(
            lambda frame: pd.Series(
                {
                    "rows": len(frame),
                    "rmse": rmse(frame["tvt_true"], frame["prediction"]),
                    **{
                        f"{family}_rmse": rmse(frame["tvt_true"], frame[f"{family}_prediction"])
                        for family in FAMILY_ORDER
                    },
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    prediction_columns = [
        "well_id",
        "fold",
        "id",
        "row_idx",
        "prediction",
        *[f"{family}_prediction" for family in FAMILY_ORDER],
        *[f"{family}_posterior_std" for family in FAMILY_ORDER],
        *[f"{family}_boundary_mass" for family in FAMILY_ORDER],
        "tvt_true",
        "prediction_freeze_sha",
    ]
    oof_path = artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    runtime_path = artifacts / f"{OUTPUT_PREFIX}_runtime.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    fold_path = artifacts / f"{OUTPUT_PREFIX}_fold_metrics.csv"
    manifest_path = artifacts / f"{OUTPUT_PREFIX}_reconstruction_manifest.json"
    oof[prediction_columns].to_csv(oof_path, index=False, compression="gzip")
    runtime.to_csv(runtime_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "late_submit": True,
        "method_fidelity": "proxy",
        "source_code_publicly_available": False,
        "config_sha256": mapping_sha256(config),
        "group_assignment": run["group_meta"],
        "fold_assignment_sha256": mapping_sha256(run["fold_map"]),
        "family_order": list(FAMILY_ORDER),
        "family_weights": get_nested(config, "model.family_weights"),
        "target_wells": len(run["target_wells"]),
        "sibling_source_rows": run["sibling_row_count"],
        "truth_rows_accessed_before_prediction_freeze": run[
            "truth_rows_accessed_before_prediction_freeze"
        ],
        "trained_models": 0,
        "lightgbm_configs": 0,
        "boosters": 0,
        "pf_runs": 0,
        "beam_runs": 0,
    }
    write_json(manifest_path, manifest)
    oof_info = inspect_gzip_csv(oof_path)
    technical = {
        "family_count": len(FAMILY_ORDER),
        "family_weight_sum": sum(float(value) for value in get_nested(config, "model.family_weights").values()),
        "rows": len(oof),
        "wells": int(oof["well_id"].nunique()),
        "duplicate_ids": int(oof["id"].duplicated().sum()),
        "non_finite_predictions": int((~np.isfinite(oof["prediction"])).sum()),
        "truth_rows_accessed_before_prediction_freeze": run[
            "truth_rows_accessed_before_prediction_freeze"
        ],
        "posterior_normalization_max_abs_error": float(runtime["normalization_error_max"].max()),
        "boundary_mass_max": float(runtime["boundary_mass_max"].max()),
        "all_validation_wells_excluded_from_atlas": True,
    }
    technical["passed"] = bool(
        technical["family_count"] == 3
        and math.isclose(technical["family_weight_sum"], 1.0, abs_tol=1e-12)
        and technical["duplicate_ids"] == 0
        and technical["non_finite_predictions"] == 0
        and technical["truth_rows_accessed_before_prediction_freeze"] == 0
        and technical["posterior_normalization_max_abs_error"] <= float(
            get_nested(config, "guards.technical.require_posterior_normalization_max_abs_error")
        )
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "debug_smoke_complete" if run["debug"] else "train_oof_complete_late_submit_pending",
        "method_fidelity": "proxy",
        "submission_phase": "post_competition_late_submission",
        "debug": run["debug"],
        "cv": pooled,
        "family_rmse": family_rmse,
        "folds": fold_rows,
        "public_lb": None,
        "private_lb": None,
        "runtime_seconds": run["elapsed_seconds"],
        "technical_gate": technical,
        "artifacts": {
            "oof": oof_info,
            "runtime_sha256": sha256_path(runtime_path),
            "by_well_sha256": sha256_path(by_well_path),
            "fold_metrics_sha256": sha256_path(fold_path),
            "reconstruction_manifest_sha256": sha256_path(manifest_path),
        },
        "external_reference": {
            "third_place_three_family_oof_rmse": 5.9703,
            "third_place_public_lb": 6.207,
            "third_place_private_lb": 6.229,
        },
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True), flush=True)
    if not technical["passed"]:
        raise RuntimeError(f"technical gate failed: {technical}")
    return metrics


def run_train(config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    run = run_oof(config)
    metrics = summarize_and_write(run, config)
    print(
        f"EXP515 LATE SUBMIT RECONSTRUCTION TRAIN COMPLETE elapsed={time.time() - started:.3f}s",
        flush=True,
    )
    return metrics


# %% [markdown]
# ## 8. Setup preview and Kaggle CPU execution

# %%
CONFIG = load_experiment_config()
validate_frozen_contract(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "late_submit": True,
            "phase": get_nested(CONFIG, "late_submission.phase"),
            "method_fidelity": get_nested(CONFIG, "method_fidelity.classification"),
            "families": list(FAMILY_ORDER),
            "weights": get_nested(CONFIG, "model.family_weights"),
            "execution_contract": get_nested(CONFIG, "execution_contract"),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "numba_available": NUMBA_AVAILABLE,
                "numba_threads_before_set": get_num_threads() if NUMBA_AVAILABLE else None,
            },
        },
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)

# %%
if EXECUTE_NOTEBOOK:
    TRAIN_METRICS = run_train(CONFIG)

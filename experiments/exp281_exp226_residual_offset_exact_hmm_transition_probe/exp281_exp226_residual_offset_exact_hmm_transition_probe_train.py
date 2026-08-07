# %% [markdown]
# # exp281 exp226 residual-offset exact HMM transition probe
#
# Group-safe exp226 `tvt_geop` is the moving coordinate center. The exact HMM
# estimates only a slow vertical offset and its minimal rate state. Raw-GR
# emission follows exp209, and truth is attached only after all paths freeze.

# %% [markdown]
# ## Contents
# 1. Imports and experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Fixed parent-cache loading and alignment
# 4. Raw-well and known-prefix HMM input preparation
# 5. Exact exp209 forward-backward kernel
# 6. Exp226 residual-offset HMM generation
# 7. Metrics and persistent-offset recovery diagnostics
# 8. Full Kaggle CPU orchestration and artifact guards
# 9. Setup and input preflight
# 10. Run generation and report artifacts

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import time
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
except ImportError:  # pragma: no cover - Kaggle includes numba.
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


EXPERIMENT_NAME = "exp281_exp226_residual_offset_exact_hmm_transition_probe"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
CANDIDATES = (
    "exp226_pred",
    "exact_hmm",
    "exp263_fixed",
    "residual_offset_hmm",
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP281_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    raise FileNotFoundError(f"exp281 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def train_data_dir(config: dict[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
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


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for column in columns:
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


def resolve_source(config: dict[str, Any], key: str) -> tuple[Path, str, str]:
    spec = get_nested(config, f"data.{key}") or {}
    path = resolve_existing(
        str(spec["filename"]), [str(value) for value in spec.get("candidates", [])]
    )
    raw_sha = sha256_path(path)
    decompressed_sha = sha256_gzip_decompressed(path)
    expected = str(spec.get("expected_decompressed_sha256") or "")
    if expected and decompressed_sha != expected:
        raise ValueError(
            f"{key} decompressed SHA mismatch: expected={expected} actual={decompressed_sha}"
        )
    return path, raw_sha, decompressed_sha


def validate_scientific_contract(config: dict[str, Any]) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp281 route must be pf_beam")
    if get_nested(config, "lineage.parent") != (
        "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
    ):
        raise ValueError("exp281 shape parent must be exp226")
    if get_nested(config, "lineage.decoder_parent") != (
        "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    ):
        raise ValueError("exp281 decoder parent must be exp209")
    if get_nested(config, "lineage.separability_parent") != (
        "exp280_exp226_shift_likelihood_separability_readout"
    ):
        raise ValueError("exp281 requires the passed exp280 separability parent")
    hmm = get_nested(config, "model.hmm") or {}
    fixed = {
        "delta_min_ft": -80.0,
        "delta_max_ft": 80.0,
        "step": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "emission": "gauss",
        "lam": 1.0,
        "sigma_mode": "std",
        "start_delta_ft": 0.0,
        "start_sig": 0.75,
        "initial_offset_rate": 0.0,
        "r0_sig": 0.01,
        "mom": 0.998,
        "rate_center": "zero",
        "gr_log_likelihood_clip": 600.0,
        "typewell_extension_ft": 40.0,
        "transition_center": "exp226_tvt_geop_row_delta",
    }
    for key, value in fixed.items():
        if hmm.get(key) != value:
            raise ValueError(f"exp209 HMM contract changed at {key}: {hmm.get(key)} != {value}")
    if list(hmm.get("active_variants") or []) != [
        "residual_offset_delta80_step035_rate41"
    ]:
        raise ValueError("exp281 must run the one fixed residual-offset variant")
    execution = get_nested(config, "execution") or {}
    required_zero = (
        "lightgbm_config_count",
        "trained_fold_count",
        "total_boosters",
    )
    if any(int(execution.get(key, -1)) != 0 for key in required_zero):
        raise ValueError("LightGBM configs, trained folds, and boosters must all be zero")
    if int(execution.get("active_hmm_variants", 0)) != 1:
        raise ValueError("exp281 must run exactly one HMM variant")
    forbidden = (
        bool(execution.get("control_or_parent_retraining")),
        bool(execution.get("cpu_exact_hmm_control_regenerated")),
        bool(execution.get("gpu")),
        bool(execution.get("inference")),
        bool(execution.get("submission")),
        bool(get_nested(config, "audit.persist_full_posterior")),
        bool(get_nested(config, "audit.persist_oracle_predictions")),
        bool(get_nested(config, "audit.persist_selector")),
        bool(get_nested(config, "audit.persist_candidate_blend")),
    )
    if any(forbidden):
        raise ValueError(
            "parent regeneration, GPU, inference, submission, and oracle output are forbidden"
        )


# %% [markdown]
# ## 3. Fixed parent-cache loading and alignment


# %%
def load_exp226_geometry(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, raw_sha, decompressed_sha = resolve_source(config, "exp226_oof")
    frame = pd.read_csv(
        path,
        usecols=["well_id", "row_idx", "suffix_offset", "tvt_geop", "fold"],
        dtype={"well_id": str},
    )
    frame = frame.rename(columns={"well_id": "well"})
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int32)
    frame["id"] = frame["well"] + "_" + frame["row_idx"].astype(str)
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(frame) != expected_rows or frame["well"].nunique() != expected_wells:
        raise ValueError(
            f"exp226 coverage mismatch rows={len(frame)}/{expected_rows} "
            f"wells={frame['well'].nunique()}/{expected_wells}"
        )
    if frame["id"].duplicated().any():
        raise ValueError("exp226 OOF has duplicate ids")
    numeric = frame[["row_idx", "suffix_offset", "tvt_geop", "fold"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("exp226 OOF contains non-finite required values")
    fold_per_well = frame.groupby("well", sort=False)["fold"].nunique()
    if not bool((fold_per_well == 1).all()):
        raise ValueError("exp226 group-safe contract failed: a well spans multiple folds")
    observed_folds = sorted(frame["fold"].astype(int).unique().tolist())
    if observed_folds != list(get_nested(config, "validation.expected_folds")):
        raise ValueError(f"exp226 fold set mismatch: {observed_folds}")
    manifest = {
        "source": "exp226_oof",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "folds": observed_folds,
    }
    return frame, manifest


def strict_merge_controls(
    generated: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    exp226_path, exp226_raw, exp226_decompressed = resolve_source(config, "exp226_oof")
    exp226_control = pd.read_csv(
        exp226_path,
        usecols=["well_id", "row_idx", "tvt_true", "tvt_pred"],
        dtype={"well_id": str},
    ).rename(
        columns={
            "well_id": "well",
            "tvt_true": "true_tvt_readout_only",
            "tvt_pred": "exp226_pred",
        }
    )
    exp226_control["row_idx"] = pd.to_numeric(
        exp226_control["row_idx"], errors="raise"
    ).astype(np.int32)
    exp226_control["id"] = (
        exp226_control["well"].astype(str) + "_" + exp226_control["row_idx"].astype(str)
    )
    exp209_path, exp209_raw, exp209_decompressed = resolve_source(config, "exp209_oof")
    exact = pd.read_csv(
        exp209_path,
        usecols=["id", "well", "hmm_mean_tvt"],
        dtype={"id": str, "well": str},
    ).rename(columns={"hmm_mean_tvt": "exact_hmm"})
    exp072_path, exp072_raw, exp072_decompressed = resolve_source(config, "exp072_oof")
    likelihood_pf = pd.read_csv(
        exp072_path,
        usecols=["id", "well", "last_known_tvt", "md_since", "likpf_mean_d"],
        dtype={"id": str, "well": str},
    )
    likelihood_pf["likelihood_pf_mean"] = pd.to_numeric(
        likelihood_pf["last_known_tvt"], errors="raise"
    ) + pd.to_numeric(likelihood_pf["likpf_mean_d"], errors="raise")
    likelihood_pf = likelihood_pf.drop(columns=["last_known_tvt", "likpf_mean_d"])
    for name, frame in (
        ("exp226_truth_control", exp226_control),
        ("exp209", exact),
        ("exp072", likelihood_pf),
    ):
        if frame["id"].duplicated().any():
            raise ValueError(f"{name} control contains duplicate ids")
        if len(frame) != len(generated):
            raise ValueError(f"{name} row count {len(frame)} != generated {len(generated)}")
    merged = generated.merge(
        exp226_control,
        on=["id", "well", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(exact, on=["id", "well"], how="left", validate="one_to_one")
    merged = merged.merge(likelihood_pf, on=["id", "well"], how="left", validate="one_to_one")
    required = [
        "true_tvt_readout_only",
        "exp226_pred",
        "exact_hmm",
        "likelihood_pf_mean",
        "md_since",
    ]
    if not np.isfinite(merged[required].to_numpy(np.float64)).all():
        raise ValueError("saved control alignment produced missing or non-finite values")
    merged["exp263_fixed"] = (
        np.float32(0.50) * merged["exp226_pred"].to_numpy(np.float32)
        + np.float32(0.25) * merged["likelihood_pf_mean"].to_numpy(np.float32)
        + np.float32(0.25) * merged["exact_hmm"].to_numpy(np.float32)
    ).astype(np.float64)
    manifests = [
        {
            "source": "exp226_truth_control_post_freeze",
            "path": str(exp226_path),
            "bytes": exp226_path.stat().st_size,
            "raw_sha256": exp226_raw,
            "decompressed_sha256": exp226_decompressed,
            "rows": len(exp226_control),
            "wells": int(exp226_control["well"].nunique()),
        },
        {
            "source": "exp209_oof",
            "path": str(exp209_path),
            "bytes": exp209_path.stat().st_size,
            "raw_sha256": exp209_raw,
            "decompressed_sha256": exp209_decompressed,
            "rows": len(exact),
            "wells": int(exact["well"].nunique()),
        },
        {
            "source": "exp072_oof",
            "path": str(exp072_path),
            "bytes": exp072_path.stat().st_size,
            "raw_sha256": exp072_raw,
            "decompressed_sha256": exp072_decompressed,
            "rows": len(likelihood_pf),
            "wells": int(likelihood_pf["well"].nunique()),
        },
    ]
    return merged, manifests


# %% [markdown]
# ## 4. Raw-well and known-prefix HMM input preparation


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
    horizontal = pd.read_csv(
        root / f"{well}__horizontal_well.csv",
        usecols=lambda column: column != "TVT",
    )
    typewell = pd.read_csv(root / f"{well}__typewell.csv").sort_values("TVT").reset_index(drop=True)
    if "TVT" in horizontal.columns:
        raise AssertionError("candidate-generation horizontal unexpectedly contains true TVT")
    return horizontal, typewell


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
    dtvt, dz, dmd = np.diff(tvt), np.diff(z), np.diff(md)
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
    return float(cal_a), float(cal_b), sigma, init_rate, effective_rows, valid_steps


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    geop_tvt: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal.columns:
        raise ValueError("prepare_hmm_inputs forbids unknown-suffix true TVT")
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal.columns):
        raise ValueError(
            f"horizontal missing {sorted(required_horizontal - set(horizontal.columns))}"
        )
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
    hmm = get_nested(config, "model.hmm") or {}
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or len(eval_rows) == 0:
        raise ValueError("well requires at least four known rows and one evaluation row")
    geop = np.asarray(geop_tvt, dtype=np.float64)
    if len(geop) != len(eval_rows) or not np.isfinite(geop).all():
        raise ValueError("exp226 tvt_geop must be finite and align to every evaluation row")

    cal_a, cal_b, robust_sigma, init_rate, rate_rows, valid_steps = prefix_stats(
        horizontal, typewell_tvt, typewell_gr, tail_n=30
    )
    if hmm["sigma_mode"] == "std":
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
    step = float(hmm["step"])
    delta_min = float(hmm["delta_min_ft"])
    delta_max = float(hmm["delta_max_ft"])
    if not delta_min < 0.0 < delta_max:
        raise ValueError("residual-offset grid must straddle zero")
    grid = np.arange(delta_min, delta_max + 0.5 * step, step, dtype=np.float64)
    if len(grid) < 3:
        raise ValueError("residual-offset grid is too small")

    md = eval_rows["MD"].to_numpy(np.float64)
    gr_fill = float(np.nanmean(typewell_gr))
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[eval_rows.index]
    )
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    absolute_tvt_states = geop[:, None] + grid[None, :]
    gr_grid = cal_a_use * np.interp(
        absolute_tvt_states, typewell_tvt, typewell_gr
    ) + cal_b_use
    zscore = (gr[:, None] - gr_grid) / gr_sigma
    if hmm["emission"] != "gauss":
        raise ValueError("exp281 fixes the exp209 Gaussian GR emission")
    emission_ll = (
        -0.5
        * np.minimum(zscore**2, float(hmm["gr_log_likelihood_clip"]))
    ).astype(np.float32)

    if hmm["rate_center"] != "zero":
        raise ValueError("exp281 offset-rate states must be centered at zero")
    rates = np.linspace(
        -float(hmm["rate_span"]),
        float(hmm["rate_span"]),
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    start_delta = float(hmm["start_delta_ft"])
    zero_quantization_error = float(np.min(np.abs(grid - start_delta)))
    native_typewell = (absolute_tvt_states >= float(typewell_tvt.min())) & (
        absolute_tvt_states <= float(typewell_tvt.max())
    )
    return {
        "emission_ll": emission_ll,
        "dm": dm,
        "grid": grid,
        "rates": rates,
        "start_p": float((start_delta - grid[0]) / step),
        "r0": float(hmm["initial_offset_rate"]),
        "eval_index": eval_rows.index.to_numpy(np.int64),
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir_diagnostic_only": init_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "grid_min": float(grid[0]),
        "grid_max": float(grid[-1]),
        "delta_zero_quantization_error_ft": zero_quantization_error,
        "delta_grid_coverage_rows": int(len(geop)),
        "delta_grid_rows": int(len(geop)),
        "native_typewell_state_coverage": float(native_typewell.mean()),
    }


# %% [markdown]
# ## 5. Exact exp209 forward-backward kernel


# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb(
    em,
    dm,
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
        for r2 in prange(r_count):
            # Position is residual offset. exp226's row-to-row TVT increment is
            # already carried by the moving coordinate center, so only the
            # slow offset-rate contributes to the delta transition.
            mu = rates[r2] * dm[t_i]
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
            mu = rates[r2] * dm[t_i]
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
# ## 6. Exp226 residual-offset HMM generation


# %%
def run_residual_offset_hmm(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    geop_tvt: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    prepared = prepare_hmm_inputs(horizontal_without_truth, typewell, geop_tvt, config)
    hmm = get_nested(config, "model.hmm") or {}
    posterior, loglik = _hmm2_fb(
        prepared["emission_ll"],
        prepared["dm"].astype(np.float64),
        float(hmm["step"]),
        prepared["rates"],
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["r0"]),
        float(hmm["r0_sig"]),
        float(hmm["lam"]),
        float(hmm["mom"]),
    )
    grid = prepared["grid"]
    delta_mean = posterior @ grid
    variance = posterior @ (grid**2) - delta_mean**2
    std = np.sqrt(np.maximum(variance, 0.0))
    del posterior
    gc.collect()
    return {
        **prepared,
        "delta_mean": np.asarray(delta_mean, dtype=np.float64),
        "mean": np.asarray(geop_tvt, dtype=np.float64) + np.asarray(
            delta_mean, dtype=np.float64
        ),
        "std": np.asarray(std, dtype=np.float64),
        "loglik": float(loglik),
    }


def build_candidate_rows_for_well(
    well: str,
    data_dir: Path,
    exp226_well: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizontal_path = data_dir / f"{well}__horizontal_well.csv"
    typewell_path = data_dir / f"{well}__typewell.csv"
    generation_horizontal, typewell = load_well(well, data_dir)
    eval_index = generation_horizontal.index[
        generation_horizontal["TVT_input"].isna()
    ].to_numpy(np.int64)
    source = exp226_well.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(eval_index, source["row_idx"].to_numpy(np.int64)):
        raise ValueError(f"exp226 row alignment failed for well={well}")
    geop = source["tvt_geop"].to_numpy(np.float64)
    started = time.time()
    result = run_residual_offset_hmm(generation_horizontal, typewell, geop, config)
    frame = pd.DataFrame(
        {
            "id": source["id"].astype(str).to_numpy(),
            "well": str(well),
            "row_idx": eval_index.astype(np.int32),
            "fold": source["fold"].to_numpy(np.int8),
            "tvt_geop": geop,
            "residual_offset_hmm": result["mean"],
            "residual_offset_delta_mean": result["delta_mean"].astype(np.float32),
            "residual_offset_hmm_std": result["std"].astype(np.float32),
            "residual_offset_hmm_loglik": np.float64(result["loglik"]),
        }
    )
    finite = np.isfinite(
        frame[
            [
                "tvt_geop",
                "residual_offset_hmm",
                "residual_offset_delta_mean",
                "residual_offset_hmm_std",
            ]
        ].to_numpy(np.float64)
    ).all()
    meta = {
        "well": str(well),
        "rows": len(frame),
        "fold": int(frame["fold"].iloc[0]),
        "status": "ok" if finite else "non_finite",
        "prefix_rows": int(result["prefix_rows"]),
        "grid_min": float(result["grid_min"]),
        "grid_max": float(result["grid_max"]),
        "grid_size": int(len(result["grid"])),
        "delta_grid_coverage_rows": int(result["delta_grid_coverage_rows"]),
        "delta_grid_rows": int(result["delta_grid_rows"]),
        "delta_grid_coverage": float(
            result["delta_grid_coverage_rows"] / result["delta_grid_rows"]
        ),
        "delta_zero_quantization_error_ft": float(
            result["delta_zero_quantization_error_ft"]
        ),
        "native_typewell_state_coverage": float(result["native_typewell_state_coverage"]),
        "delta_mean_abs_median": float(np.median(np.abs(result["delta_mean"]))),
        "delta_mean_abs_max": float(np.max(np.abs(result["delta_mean"]))),
        "posterior_std_mean": float(np.mean(result["std"])),
        "posterior_std_p90": float(np.quantile(result["std"], 0.90)),
        "loglik": float(result["loglik"]),
        "elapsed_seconds": float(time.time() - started),
        "horizontal_sha256": sha256_path(horizontal_path),
        "typewell_sha256": sha256_path(typewell_path),
    }
    return frame, meta


# %% [markdown]
# ## 7. Metrics and persistent-offset recovery diagnostics


# %%
def score_prediction(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    return {
        "rows": len(error),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "within5": float(np.mean(np.abs(error) <= 5.0)),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


def candidate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    truth = frame["true_tvt_readout_only"].to_numpy(np.float64)
    rows = []
    for candidate in CANDIDATES:
        rows.append({"candidate": candidate, **score_prediction(truth, frame[candidate])})
    return pd.DataFrame(rows)


def fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in frame.groupby("fold", sort=True):
        truth = group["true_tvt_readout_only"].to_numpy(np.float64)
        for candidate in CANDIDATES:
            rows.append(
                {
                    "fold": int(fold),
                    "candidate": candidate,
                    **score_prediction(truth, group[candidate]),
                }
            )
    return pd.DataFrame(rows)


def distance_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    boundaries = [
        0.0,
        *[float(value) for value in get_nested(config, "audit.distance_buckets_ft")],
        np.inf,
    ]
    labels = []
    for index in range(len(boundaries) - 1):
        upper = boundaries[index + 1]
        upper_label = "inf" if not np.isfinite(upper) else f"{int(upper):04d}"
        labels.append(f"{int(boundaries[index]):04d}_{upper_label}")
    bucket = pd.cut(frame["md_since"], boundaries, labels=labels, right=False)
    rows = []
    for label in labels:
        group = frame.loc[bucket == label]
        if group.empty:
            continue
        truth = group["true_tvt_readout_only"].to_numpy(np.float64)
        for candidate in CANDIDATES:
            rows.append(
                {
                    "distance_bucket": label,
                    "candidate": candidate,
                    **score_prediction(truth, group[candidate]),
                }
            )
    return pd.DataFrame(rows)


def by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, group in frame.groupby("well", sort=True):
        truth = group["true_tvt_readout_only"].to_numpy(np.float64)
        baseline = score_prediction(truth, group["exp263_fixed"])["rmse"]
        for candidate in CANDIDATES:
            metric = score_prediction(truth, group[candidate])
            rows.append(
                {
                    "well": str(well),
                    "fold": int(group["fold"].iloc[0]),
                    "candidate": candidate,
                    **metric,
                    "delta_rmse_vs_exp263_fixed": float(metric["rmse"] - baseline),
                }
            )
    return pd.DataFrame(rows)


def hidden_like_metrics(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    hidden = get_nested(config, "data.hidden_like") or {}
    if not bool(hidden.get("enabled")):
        return pd.DataFrame(), None
    candidates = [str(value) for value in hidden.get("fold_assignment_candidates", [])]
    path = resolve_existing(Path(candidates[0]).name, candidates)
    assignment_sha = sha256_path(path)
    expected_sha = str(hidden.get("expected_sha256") or "")
    if expected_sha and assignment_sha != expected_sha:
        raise ValueError(
            f"hidden-like assignment SHA mismatch: {assignment_sha} != {expected_sha}"
        )
    assignments = pd.read_csv(path, dtype={"well_id": str})
    rows = []
    for subgroup, role_column in (hidden.get("valid_role_columns") or {}).items():
        if role_column not in assignments.columns:
            raise ValueError(f"hidden-like assignment missing {role_column}")
        valid_wells = set(
            assignments.loc[assignments[role_column].astype(str) == "valid", "well_id"].astype(str)
        )
        group = frame.loc[frame["well"].astype(str).isin(valid_wells)]
        if group.empty:
            raise ValueError(f"hidden-like subgroup {subgroup} selected zero rows")
        truth = group["true_tvt_readout_only"].to_numpy(np.float64)
        for candidate in CANDIDATES:
            rows.append(
                {
                    "subgroup": str(subgroup),
                    "candidate": candidate,
                    **score_prediction(truth, group[candidate]),
                }
            )
    manifest = {
        "source": "hidden_like_assignments",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": assignment_sha,
        "rows": len(assignments),
        "wells": int(assignments["well_id"].nunique()),
    }
    return pd.DataFrame(rows), manifest


def persistent_offset_episodes(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = get_nested(config, "audit.persistent_offset") or {}
    threshold = float(spec["error_threshold_ft"])
    minimum_rows = int(spec["minimum_consecutive_rows"])
    return_threshold = float(spec["return_threshold_ft"])
    horizons = [int(value) for value in spec["recovery_horizons_rows"]]
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for well, group in frame.groupby("well", sort=True):
            group = group.sort_values("row_idx", kind="mergesort")
            error = np.abs(
                group[candidate].to_numpy(np.float64)
                - group["true_tvt_readout_only"].to_numpy(np.float64)
            )
            bad = error > threshold
            padded = np.concatenate([[False], bad, [False]])
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends = np.flatnonzero(padded[:-1] & ~padded[1:])
            row_index = group["row_idx"].to_numpy(np.int64)
            for start, end in zip(starts, ends, strict=True):
                if end - start < minimum_rows:
                    continue
                confirmed_at = start + minimum_rows - 1
                recovery = np.flatnonzero(error[confirmed_at + 1 :] <= return_threshold)
                recovery_rows = int(recovery[0] + 1) if len(recovery) else None
                row = {
                    "candidate": candidate,
                    "well": str(well),
                    "fold": int(group["fold"].iloc[0]),
                    "episode_start_row_idx": int(row_index[start]),
                    "confirmed_row_idx": int(row_index[confirmed_at]),
                    "consecutive_rows_above_threshold": int(end - start),
                    "peak_abs_error_ft": float(np.max(error[start:end])),
                    "recovery_rows_after_confirmation": recovery_rows,
                }
                for horizon in horizons:
                    row[f"recovered_within_{horizon}"] = bool(
                        recovery_rows is not None and recovery_rows <= horizon
                    )
                rows.append(row)
    episodes = pd.DataFrame(rows)
    summary_rows = []
    for candidate in CANDIDATES:
        group = episodes.loc[episodes["candidate"] == candidate] if not episodes.empty else episodes
        row: dict[str, Any] = {"candidate": candidate, "episodes": len(group)}
        for horizon in horizons:
            column = f"recovered_within_{horizon}"
            row[column + "_count"] = int(group[column].sum()) if len(group) else 0
            row[column + "_rate"] = float(group[column].mean()) if len(group) else np.nan
        summary_rows.append(row)
    return episodes, pd.DataFrame(summary_rows)


def scope_rmse(frame: pd.DataFrame, candidate: str, mask: np.ndarray) -> float:
    group = frame.loc[mask]
    return score_prediction(group["true_tvt_readout_only"], group[candidate])["rmse"]


def evaluate_promotion_guard(
    frame: pd.DataFrame,
    overall: pd.DataFrame,
    folds: pd.DataFrame,
    hidden: pd.DataFrame,
    by_well: pd.DataFrame,
    well_manifest: pd.DataFrame,
    recovery: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.guards") or {}
    candidate_rmse = float(
        overall.loc[overall["candidate"] == "residual_offset_hmm", "rmse"].iloc[0]
    )
    baseline_rmse = float(overall.loc[overall["candidate"] == "exp263_fixed", "rmse"].iloc[0])
    expected_baseline_rmse = float(get_nested(config, "validation.promotion_baseline.oof_rmse"))
    baseline_parity_atol = float(get_nested(config, "validation.promotion_baseline.parity_atol_ft"))
    baseline_parity_abs_diff = abs(baseline_rmse - expected_baseline_rmse)
    minimum_gain = float(guards["minimum_oof_gain_ft"])
    fold_wide = folds.pivot(index="fold", columns="candidate", values="rmse")
    improved_folds = int((fold_wide["residual_offset_hmm"] < fold_wide["exp263_fixed"]).sum())
    md_since = frame["md_since"].to_numpy(np.float64)
    scope_masks = {
        "near": (md_since >= 0.0) & (md_since < 250.0),
        "long_tail": md_since >= 1000.0,
    }
    scope_deltas: dict[str, float] = {}
    for name, mask in scope_masks.items():
        scope_deltas[name] = scope_rmse(frame, "residual_offset_hmm", mask) - scope_rmse(
            frame, "exp263_fixed", mask
        )
    if not hidden.empty:
        hidden_wide = hidden.pivot(index="subgroup", columns="candidate", values="rmse")
        for subgroup in hidden_wide.index:
            scope_deltas[f"hidden_like::{subgroup}"] = float(
                hidden_wide.loc[subgroup, "residual_offset_hmm"]
                - hidden_wide.loc[subgroup, "exp263_fixed"]
            )
    maximum_scope_regression = float(guards["maximum_scope_rmse_regression_ft"])
    worst_well = float(
        by_well.loc[
            by_well["candidate"] == "residual_offset_hmm",
            "delta_rmse_vs_exp263_fixed",
        ].max()
    )
    delta_grid_coverage = float(
        well_manifest["delta_grid_coverage_rows"].sum()
        / well_manifest["delta_grid_rows"].sum()
    )
    finite_coverage = float(np.isfinite(frame[list(CANDIDATES)].to_numpy(np.float64)).mean())
    recovery_wide = recovery.set_index("candidate")
    candidate_episodes = int(recovery_wide.loc["residual_offset_hmm", "episodes"])
    baseline_episodes = int(recovery_wide.loc["exp263_fixed", "episodes"])
    episode_delta = candidate_episodes - baseline_episodes
    recovery_rate_deltas: dict[str, float] = {}
    recovery_checks: dict[str, bool] = {}
    minimum_recovery_delta = float(guards["minimum_recovery_rate_delta"])
    for horizon in get_nested(config, "audit.persistent_offset.recovery_horizons_rows"):
        column = f"recovered_within_{int(horizon)}_rate"
        candidate_rate = float(recovery_wide.loc["residual_offset_hmm", column])
        baseline_rate = float(recovery_wide.loc["exp263_fixed", column])
        delta = candidate_rate - baseline_rate
        recovery_rate_deltas[str(int(horizon))] = delta
        recovery_checks[str(int(horizon))] = bool(
            candidate_episodes == 0 or delta >= minimum_recovery_delta
        )
    checks = {
        "exp263_fixed_parity": bool(baseline_parity_abs_diff <= baseline_parity_atol),
        "overall_gain": bool(baseline_rmse - candidate_rmse >= minimum_gain),
        "improved_folds": bool(improved_folds >= int(guards["minimum_improved_folds"])),
        "scope_regression": bool(
            all(delta <= maximum_scope_regression for delta in scope_deltas.values())
        ),
        "worst_well": bool(worst_well <= float(guards["maximum_worst_well_rmse_regression_ft"])),
        "persistent_episode_count": bool(
            episode_delta <= int(guards["maximum_persistent_episode_count_delta"])
        ),
        "persistent_recovery_rates": bool(all(recovery_checks.values())),
        "delta_grid_coverage": bool(
            delta_grid_coverage >= float(guards["required_delta_grid_coverage"])
        ),
        "finite_coverage": bool(finite_coverage >= float(guards["required_finite_coverage"])),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_rmse": candidate_rmse,
        "baseline_rmse": baseline_rmse,
        "expected_baseline_rmse": expected_baseline_rmse,
        "baseline_parity_abs_diff_ft": baseline_parity_abs_diff,
        "baseline_parity_atol_ft": baseline_parity_atol,
        "gain_ft": baseline_rmse - candidate_rmse,
        "improved_folds": improved_folds,
        "scope_delta_rmse_vs_exp263_fixed": scope_deltas,
        "worst_well_delta_rmse_vs_exp263_fixed": worst_well,
        "persistent_episode_count": {
            "candidate": candidate_episodes,
            "baseline": baseline_episodes,
            "delta": episode_delta,
        },
        "persistent_recovery_rate_delta": recovery_rate_deltas,
        "persistent_recovery_checks": recovery_checks,
        "delta_grid_coverage": delta_grid_coverage,
        "finite_coverage": finite_coverage,
    }


# %% [markdown]
# ## 8. Full Kaggle CPU orchestration and artifact guards


# %%
def run_full_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp281 generation must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is "
            "reserved for an explicitly approved local smoke run."
        )
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("exp281 Kaggle CPU execution is not approved")
    validate_scientific_contract(config)
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for exact HMM generation")
    set_num_threads(int(get_nested(config, "execution.numba_num_threads") or 1))
    started = time.time()
    exp226, exp226_manifest = load_exp226_geometry(config)
    data_dir = train_data_dir(config)
    wells = list_well_ids(data_dir)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(wells) != expected_wells or set(wells) != set(exp226["well"].unique()):
        raise ValueError("raw-well and exp226 well sets do not match")
    max_wells = get_nested(config, "execution.max_wells")
    environment_max = int(os.environ.get("EXPERIMENT_MAX_WELLS", "0") or "0")
    if environment_max:
        max_wells = environment_max
    if max_wells is not None:
        wells = wells[: int(max_wells)]
    full_run = len(wells) == expected_wells
    grouped_exp226 = {well: group for well, group in exp226.groupby("well", sort=False)}
    frames: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    progress_every = int(get_nested(config, "execution.progress_every_wells") or 1)
    for index, well in enumerate(wells, start=1):
        print(f"[exp281] {index}/{len(wells)} well={well}", flush=True)
        frame, meta = build_candidate_rows_for_well(well, data_dir, grouped_exp226[well], config)
        frames.append(frame)
        well_rows.append(meta)
        if index == 1 or index % progress_every == 0:
            print(json.dumps(to_jsonable(meta), sort_keys=True), flush=True)
        gc.collect()
    generated = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    del frames, grouped_exp226, exp226
    gc.collect()
    if full_run:
        expected_rows = int(get_nested(config, "validation.expected_rows"))
        if len(generated) != expected_rows or generated["well"].nunique() != expected_wells:
            raise ValueError("generated full-run coverage mismatch")
    if generated["id"].duplicated().any():
        raise ValueError("generated candidate contains duplicate ids")
    well_manifest = pd.DataFrame(well_rows).sort_values("well").reset_index(drop=True)
    required_grid_coverage = float(
        get_nested(config, "validation.guards.required_delta_grid_coverage")
    )
    actual_grid_coverage = float(
        well_manifest["delta_grid_coverage_rows"].sum()
        / well_manifest["delta_grid_rows"].sum()
    )
    if actual_grid_coverage < required_grid_coverage:
        raise RuntimeError(
            f"delta grid coverage {actual_grid_coverage} < required {required_grid_coverage}"
        )

    generated, control_manifests = strict_merge_controls(generated, config)
    overall = candidate_metrics(generated)
    folds = fold_metrics(generated)
    distance = distance_metrics(generated, config)
    by_well = by_well_metrics(generated)
    hidden, hidden_manifest = hidden_like_metrics(generated, config)
    episodes, recovery = persistent_offset_episodes(generated, config)
    guard = evaluate_promotion_guard(
        generated, overall, folds, hidden, by_well, well_manifest, recovery, config
    )

    artifacts = artifact_dir()
    paths = {
        "predictions": artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz",
        "candidate_metrics": artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv",
        "fold_metrics": artifacts / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "distance_metrics": artifacts / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv",
        "hidden_metrics": artifacts / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv",
        "by_well": artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "episodes": artifacts / f"{OUTPUT_PREFIX}_recovery_episodes.csv",
        "recovery": artifacts / f"{OUTPUT_PREFIX}_recovery_summary.csv",
        "well_manifest": artifacts / f"{OUTPUT_PREFIX}_well_manifest.csv",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "decoder_manifest": artifacts / f"{OUTPUT_PREFIX}_decoder_manifest.json",
        "summary": artifacts / f"{OUTPUT_PREFIX}_summary.json",
    }
    generated.to_csv(paths["predictions"], index=False, compression="gzip")
    overall.to_csv(paths["candidate_metrics"], index=False)
    folds.to_csv(paths["fold_metrics"], index=False)
    distance.to_csv(paths["distance_metrics"], index=False)
    hidden.to_csv(paths["hidden_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    episodes.to_csv(paths["episodes"], index=False)
    recovery.to_csv(paths["recovery"], index=False)
    well_manifest.to_csv(paths["well_manifest"], index=False)
    input_manifests = [exp226_manifest, *control_manifests]
    if hidden_manifest is not None:
        input_manifests.append(hidden_manifest)
    pd.DataFrame(input_manifests).to_csv(paths["input_manifest"], index=False)
    decoder_manifest = {
        "parent": get_nested(config, "lineage.parent"),
        "decoder_parent": get_nested(config, "lineage.decoder_parent"),
        "separability_parent": get_nested(config, "lineage.separability_parent"),
        "hmm": get_nested(config, "model.hmm"),
        "coordinate_contract": {
            "equation": "TVT_t = exp226_tvt_geop_t + delta_t",
            "absolute_transition_center": "diff(exp226_tvt_geop)",
            "delta_transition_center": "offset_rate_t * delta_md_t",
            "truth_available_to_decoder": False,
            "exp226_tvt_pred_available_to_decoder": False,
        },
        "fixed_formula_control": get_nested(config, "model.fixed_formula_control"),
        "truth_attachment": "after_all_well_candidate_paths_freeze",
        "candidate_count": 1,
    }
    write_json(paths["decoder_manifest"], decoder_manifest)
    prediction_sha = dataframe_content_sha(
        generated,
        [
            "id",
            "row_idx",
            "fold",
            "tvt_geop",
            "residual_offset_delta_mean",
            "residual_offset_hmm",
            "residual_offset_hmm_std",
        ],
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_guard_passed"
        if guard["passed"]
        else "completed_train_side_guard_failed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(generated),
        "wells": int(generated["well"].nunique()),
        "full_run": full_run,
        "active_hmm_variants": 1,
        "hmm_well_runs": len(wells),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "gpu": False,
        "inference": False,
        "submission": False,
        "elapsed_seconds": float(time.time() - started),
        "promotion_guard": guard,
        "candidate_metrics": overall.to_dict("records"),
        "recovery_summary": recovery.to_dict("records"),
        "prediction_content_sha256": prediction_sha,
        "decoder_manifest_sha256": mapping_sha256(decoder_manifest),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "sha256": {
            "predictions_raw_gzip": sha256_path(paths["predictions"]),
            "predictions_decompressed": sha256_gzip_decompressed(paths["predictions"]),
            **{
                key: sha256_path(path)
                for key, path in paths.items()
                if key not in {"predictions", "summary"}
            },
        },
    }
    write_json(paths["summary"], summary)
    metrics_path = artifacts.parent / "metrics.json"
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": summary["status"],
            "metric": "rmse_tvt",
            "cv": guard["candidate_rmse"],
            "public_lb": None,
            "private_lb": None,
            "rows": len(generated),
            "wells": int(generated["well"].nunique()),
            "promotion_guard": guard,
            "candidate_metrics": overall.to_dict("records"),
            "prediction_content_sha256": prediction_sha,
            "decoder_manifest_sha256": summary["decoder_manifest_sha256"],
            "summary": str(paths["summary"]),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


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
                "active_hmm_variants": get_nested(CONFIG, "execution.active_hmm_variants"),
                "total_hmm_well_runs": get_nested(CONFIG, "execution.total_hmm_well_runs"),
                "lightgbm_configs": get_nested(CONFIG, "execution.lightgbm_config_count"),
                "trained_folds": get_nested(CONFIG, "execution.trained_fold_count"),
                "boosters": get_nested(CONFIG, "execution.total_boosters"),
                "parent_control_retraining": get_nested(
                    CONFIG, "execution.control_or_parent_retraining"
                ),
                "hmm": get_nested(CONFIG, "model.hmm"),
                "gpu": get_nested(CONFIG, "execution.gpu"),
                "inference": get_nested(CONFIG, "execution.inference"),
                "submission": get_nested(CONFIG, "execution.submission"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "numba_available": NUMBA_AVAILABLE,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


# %% [markdown]
# ## 10. Run generation and report artifacts


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_full_experiment(CONFIG)
    print(json.dumps(to_jsonable(SUMMARY), indent=2, sort_keys=True), flush=True)

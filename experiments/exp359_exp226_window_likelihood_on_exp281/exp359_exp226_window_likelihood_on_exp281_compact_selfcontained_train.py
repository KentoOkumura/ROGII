# %% [markdown]
# # exp359 exp226 window likelihood on exp281
#
# Stage 0 only: build the fixed exp226 500-row GR window potential, compare its
# shift ranking with the SHA-pinned saved exp280 Gaussian aggregate, and attach
# unknown-suffix truth only after the complete target-free bundle is frozen.
# No HMM, model, prediction correction, inference, or submission is run.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Fixed scientific contract and input loaders
# 4. Exp226 window profile and sparse-potential scoring
# 5. Saved exp280 control alignment and target-free freeze
# 6. Truth-late window rank readout
# 7. Scope metrics and frozen Stage 0 gate
# 8. Kaggle CPU Stage 0 orchestration
# 9. Setup, contract preview, and guarded execution

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp359_exp226_window_likelihood_on_exp281"
OUTPUT_PREFIX = EXPERIMENT_NAME
EXPECTED_SHIFTS = np.asarray(
    [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    dtype=np.float64,
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return False
    return shell is not None


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers

# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents, KAGGLE_WORKING_ROOT]
    for candidate in candidates:
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    for candidate in candidates:
        if (candidate / "experiments" / EXPERIMENT_NAME / "config.yaml").exists():
            return candidate
    return Path.cwd()


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = [
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return read_yaml(path)
    raise FileNotFoundError("exp359 config.yaml was not restored by the Kaggle bootstrap")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    candidates = [
        configured if configured.is_absolute() else project_root() / configured,
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
    ]
    for candidate in candidates:
        if candidate.exists() and next(candidate.glob("*__horizontal_well.csv"), None):
            return candidate
    for candidate in sorted(KAGGLE_INPUT_ROOT.rglob("train")) if KAGGLE_INPUT_ROOT.exists() else []:
        if candidate.is_dir() and next(candidate.glob("*__horizontal_well.csv"), None):
            return candidate
    raise FileNotFoundError("could not resolve the raw train well directory")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> str:
    selected = frame if columns is None else frame.loc[:, columns]
    return hashlib.sha256(selected.to_csv(index=False).encode()).hexdigest()


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False).encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(content)
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(content).hexdigest(),
        "content_sha256": dataframe_content_sha(frame),
    }


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    paths = [Path(value) for value in candidates]
    for path in paths:
        if path.exists():
            return path
    roots = [project_root(), KAGGLE_INPUT_ROOT, Path("/tmp")]
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"could not resolve {filename}; candidates={paths}")


def resolve_pattern_file(filename: str, patterns: Iterable[str]) -> Path:
    for pattern in patterns:
        if not any(token in pattern for token in "*?[]"):
            path = Path(pattern)
            if path.exists():
                return path
            continue
        roots = [project_root(), KAGGLE_INPUT_ROOT, Path("/tmp")]
        relative = pattern[3:] if pattern.startswith("**/") else pattern
        for root in roots:
            if not root.exists():
                continue
            matches = sorted(root.glob(relative))
            if matches:
                return matches[0]
    return resolve_existing(filename, [])


def stable_uint64(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def stable_score_permutation(
    well_id: str,
    window_id: int,
    profile_sha256: str,
    candidate_count: int,
) -> np.ndarray:
    if candidate_count < 2:
        raise ValueError("negative-control permutation requires at least two candidates")
    rng = np.random.default_rng(
        stable_uint64(EXPERIMENT_NAME, well_id, window_id, profile_sha256)
    )
    permutation = rng.permutation(candidate_count)
    if np.array_equal(permutation, np.arange(candidate_count)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def rank_descending(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rank scores must be a finite one-dimensional array")
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


# %% [markdown]
# ## 3. Fixed scientific contract and input loaders

# %%
def validate_scientific_contract(
    config: dict[str, Any],
    *,
    require_run_approval: bool = False,
) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp359 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != (
        "exp281_exp226_residual_offset_exact_hmm_transition_probe"
    ):
        raise ValueError("exp359 parent changed")
    shifts = np.asarray(get_nested(config, "model.window_potential.shifts_ft"), dtype=np.float64)
    if not np.array_equal(shifts, EXPECTED_SHIFTS):
        raise ValueError("fixed 13-shift bank changed")
    fixed_values = {
        "model.window_potential.window_rows": 500,
        "model.window_potential.stride_rows": 125,
        "model.window_potential.profile_grid_step_ft": 0.5,
        "model.window_potential.minimum_finite_gr_fraction": 0.5,
        "model.window_potential.minimum_profile_bins": 7,
        "model.window_potential.minimum_relative_path_span_ft": 4.0,
        "model.window_potential.state_std_floor": 1e-6,
        "data.exp280_gaussian_control.block_rows": 512,
        "execution_contract.stage_0.scientific_scores": 1,
        "execution_contract.stage_0.saved_control_scores": 1,
        "execution_contract.stage_0.hmm_well_runs": 0,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
    }
    for key, expected in fixed_values.items():
        if get_nested(config, key) != expected:
            raise ValueError(f"fixed contract changed: {key}")
    weights = get_nested(config, "model.window_potential.score_weights") or {}
    if weights != {"correlation": 2.0, "mse": 0.5, "level": 0.1}:
        raise ValueError("exp226 score weights changed")
    if get_nested(config, "model.window_potential.sigma_clip") != [5.0, 60.0]:
        raise ValueError("exp226 known-prefix sigma clip changed")
    if get_nested(config, "model.window_potential.lambda_clip") != [0.3, 1.0]:
        raise ValueError("window lambda clip changed")
    if get_nested(config, "model.window_potential.posterior_sd_source") != (
        "fixed_13_shift_softmax_of_normalized_window_scores"
    ):
        raise ValueError("posterior-SD source changed")
    if get_nested(config, "model.stage_0.control_alignment") != (
        "window_center_to_containing_exp280_512_row_block"
    ):
        raise ValueError("saved-control alignment changed")
    if bool(get_nested(config, "implementation.stage_1_implemented")):
        raise ValueError("Stage 1 is not implemented by this notebook")
    for key in ("execution.run_stage_1", "execution.run_inference", "execution.create_submission"):
        if bool(get_nested(config, key)):
            raise ValueError(f"{key} must remain false")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
        raise ValueError("Stage 0 must remain CPU-only")
    if bool(get_nested(config, "runtime.kaggle.enable_internet")):
        raise ValueError("Stage 0 must remain offline")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise PermissionError("Kaggle Stage 0 execution is not approved")
        if not bool(get_nested(config, "execution.run_stage_0")):
            raise PermissionError("execution.run_stage_0 is false")


def load_exp280_gaussian_control(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = get_nested(config, "data.exp280_gaussian_control") or {}
    score_path = resolve_pattern_file(
        str(spec["score_filename"]),
        [str(value) for value in spec["score_patterns"]],
    )
    decompressed_sha = sha256_gzip_decompressed(score_path)
    if decompressed_sha != str(spec["score_decompressed_sha256"]):
        raise ValueError("exp280 Gaussian score decompressed SHA changed")
    contract_path = resolve_pattern_file(
        str(spec["contract_filename"]),
        [str(value) for value in spec["contract_patterns"]],
    )
    contract = json.loads(contract_path.read_text())
    if bool(contract.get("truth_attached")):
        raise ValueError("exp280 Gaussian score contract must be truth-free")
    if contract.get("target_free_score_content_sha256") != str(spec["score_content_sha256"]):
        raise ValueError("exp280 Gaussian content declaration changed")
    if contract.get("scientific_contract_sha256") != str(spec["scientific_contract_sha256"]):
        raise ValueError("exp280 scientific contract SHA changed")
    if list(map(float, contract.get("shift_bank_ft", []))) != EXPECTED_SHIFTS.tolist():
        raise ValueError("exp280 shift bank changed")
    if int(contract.get("block_rows", -1)) != int(spec["block_rows"]):
        raise ValueError("exp280 block size changed")

    scores = pd.read_csv(score_path, dtype={"well_id": str})
    forbidden = {"tvt_true", "tvt_pred", "gr_delta", "error", "abs_error", "TVT"}
    leaked = sorted(forbidden.intersection(scores.columns))
    if leaked:
        raise ValueError(f"exp280 control contains truth/error columns: {leaked}")
    required = {
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "shift_slot",
        "shift_ft",
        "likelihood_mean",
        "likelihood_rank",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"exp280 control missing {missing}")
    integer_columns = (
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "shift_slot",
        "likelihood_rank",
    )
    for column in integer_columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.int64)
    for column in ("shift_ft", "likelihood_mean"):
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.float64)
    scores["well_id"] = scores["well_id"].astype(str)
    scores = scores.sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)
    expected_blocks = int(get_nested(config, "validation.expected_exp280_blocks"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(scores) != expected_blocks * len(EXPECTED_SHIFTS):
        raise ValueError("exp280 control row count changed")
    if scores["well_id"].nunique() != expected_wells:
        raise ValueError("exp280 control well count changed")
    sizes = scores.groupby(["well_id", "block_id"], sort=False).size()
    if len(sizes) != expected_blocks or not sizes.eq(len(EXPECTED_SHIFTS)).all():
        raise ValueError("each exp280 block must contain exactly 13 shifts")
    observed_shifts = scores["shift_ft"].to_numpy().reshape(-1, len(EXPECTED_SHIFTS))
    if not np.array_equal(
        observed_shifts,
        np.broadcast_to(EXPECTED_SHIFTS, observed_shifts.shape),
    ):
        raise ValueError("exp280 shift identity/order changed")
    saved_scores = scores["likelihood_mean"].to_numpy().reshape(
        -1, len(EXPECTED_SHIFTS)
    )
    saved_ranks = scores["likelihood_rank"].to_numpy().reshape(
        -1, len(EXPECTED_SHIFTS)
    )
    recomputed = np.vstack([rank_descending(row) for row in saved_scores])
    if not np.array_equal(saved_ranks, recomputed):
        raise ValueError("exp280 stored ranks do not match saved scores")
    manifests = [
        {
            "name": "exp280_saved_gaussian_scores",
            "path": str(score_path),
            "raw_sha256": sha256_path(score_path),
            "decompressed_sha256": decompressed_sha,
            "declared_content_sha256": contract["target_free_score_content_sha256"],
            "rows": len(scores),
        },
        {
            "name": "exp280_saved_gaussian_contract",
            "path": str(contract_path),
            "raw_sha256": sha256_path(contract_path),
            "scientific_contract_sha256": contract["scientific_contract_sha256"],
        },
    ]
    return scores, manifests


def load_exp226_safe(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof") or {}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    decompressed_sha = sha256_gzip_decompressed(path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA changed")
    safe_columns = [str(value) for value in spec["safe_columns"]]
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF contains duplicate identities")
    if not np.isfinite(frame["tvt_geop"]).all():
        raise ValueError("exp226 tvt_geop must be finite")
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 OOF row count changed")
    if frame["well_id"].nunique() != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("exp226 OOF well count changed")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].unique().tolist()) != expected_folds:
        raise ValueError("exp226 fold set changed")
    if not frame.groupby("well_id")["fold"].nunique().eq(1).all():
        raise ValueError("each exp226 well must belong to one reporting fold")
    manifest = {
        "name": "exp226_group_safe_oof_safe_columns",
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "safe_columns": safe_columns,
    }
    return frame, path, manifest


def load_exp226_truth(
    path: Path,
    config: dict[str, Any],
    *,
    frozen_score_content_sha256: str,
) -> pd.DataFrame:
    if not frozen_score_content_sha256:
        raise ValueError("truth attachment requires a frozen target-free score SHA")
    spec = get_nested(config, "data.exp226_oof") or {}
    truth_columns = [str(value) for value in spec["truth_columns"]]
    frame = pd.read_csv(path, usecols=truth_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("truth rows contain duplicate identities")
    if not np.isfinite(frame["tvt_true"]).all():
        raise ValueError("truth rows must be finite")
    return frame


def load_hidden_like_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    if not bool(spec.get("enabled")):
        return pd.DataFrame(), {"name": "hidden_like_disabled", "enabled": False}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA changed")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *[str(value) for value in spec["role_columns"].values()]}
    if not required.issubset(frame.columns):
        raise ValueError(f"hidden-like assignments missing {sorted(required - set(frame.columns))}")
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments require one row per well")
    return frame, {
        "name": "exp115_hidden_like_fold_assignments",
        "path": str(path),
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


def load_horizontal_without_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda column: column != "TVT")
    if "TVT" in frame.columns:
        raise ValueError("target-free horizontal reader exposed TVT")
    return frame


# %% [markdown]
# ## 4. Exp226 window profile and sparse-potential scoring

# %%
def affine_cal(
    gr_known: np.ndarray,
    typewell_gr_at_known: np.ndarray,
) -> tuple[float, float]:
    observed = np.asarray(gr_known, dtype=np.float64)
    reference = np.asarray(typewell_gr_at_known, dtype=np.float64)
    mask = np.isfinite(observed) & np.isfinite(reference)
    if int(mask.sum()) < 30:
        return 1.0, 0.0
    x = observed[mask]
    y = reference[mask]
    for _ in range(2):
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        keep = np.abs(residual) < 2.5 * (np.std(residual) + 1e-9)
        if int(keep.sum()) < 20:
            break
        x = x[keep]
        y = y[keep]
    slope, intercept = np.polyfit(x, y, 1)
    if not (0.2 < slope < 5.0):
        return 1.0, float(np.median(y - x))
    return float(slope), float(intercept)


def prepare_gr_inputs(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("window scoring forbids horizontal TVT")
    required_horizontal = {"MD", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal_without_truth.columns):
        missing = sorted(required_horizontal - set(horizontal_without_truth.columns))
        raise ValueError(
            f"horizontal missing {missing}"
        )
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell requires TVT and GR")
    tw = typewell[["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw[["TVT", "GR"]].to_numpy()).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    known = horizontal_without_truth.loc[horizontal_without_truth["TVT_input"].notna()]
    if len(known) < 4:
        raise ValueError("well requires at least four known-prefix rows")
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    slope, intercept = affine_cal(known_gr, typewell_at_known)
    calibrated_known = known_gr * slope + intercept
    residual = calibrated_known - typewell_at_known
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "model.window_potential.sigma_clip")
    ]
    finite_residual = residual[np.isfinite(residual)]
    sigma = (
        float(np.clip(np.std(finite_residual), sigma_low, sigma_high))
        if len(finite_residual) > 30
        else 30.0
    )
    lateral_raw = pd.to_numeric(horizontal_without_truth["GR"], errors="coerce")
    lateral_calibrated = (
        lateral_raw.interpolate(limit=10).to_numpy(np.float64) * slope + intercept
    )
    return {
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "lateral_gr_calibrated": lateral_calibrated,
        "affine_slope": slope,
        "affine_intercept": intercept,
        "gr_sigma": sigma,
        "known_rows": len(known),
        "known_residual_std_unclipped": (
            float(np.std(finite_residual)) if len(finite_residual) else np.nan
        ),
    }


def build_window_profile(
    window_gr: np.ndarray,
    relative_path: np.ndarray,
    *,
    grid_step_ft: float,
    minimum_finite_fraction: float,
    minimum_profile_bins: int,
    minimum_path_span_ft: float,
) -> dict[str, Any]:
    gr = np.asarray(window_gr, dtype=np.float64)
    path = np.asarray(relative_path, dtype=np.float64)
    if gr.ndim != 1 or path.ndim != 1 or len(gr) != len(path):
        raise ValueError("window GR and relative path must be aligned one-dimensional arrays")
    finite = np.isfinite(gr)
    finite_fraction = float(finite.mean()) if len(gr) else 0.0
    centered_path = path - float(np.mean(path))
    path_span = float(np.ptp(centered_path)) if len(path) else 0.0
    base = {
        "eligible": False,
        "finite_fraction": finite_fraction,
        "path_span_ft": path_span,
        "relative_bins_ft": np.asarray([], dtype=np.float64),
        "profile_gr": np.asarray([], dtype=np.float64),
        "profile_bin_count": 0,
    }
    if finite_fraction < minimum_finite_fraction or path_span < minimum_path_span_ft:
        return base
    relative_bins = np.arange(
        float(centered_path.min()),
        float(centered_path.max()) + grid_step_ft,
        grid_step_ft,
        dtype=np.float64,
    )
    indices = np.clip(
        ((centered_path - centered_path.min()) / grid_step_ft).astype(np.int64),
        0,
        len(relative_bins) - 1,
    )
    profile = np.full(len(relative_bins), np.nan, dtype=np.float64)
    for bin_index in range(len(relative_bins)):
        values = gr[(indices == bin_index) & finite]
        if len(values) >= 3:
            profile[bin_index] = float(np.mean(values))
    ok = np.isfinite(profile)
    if int(ok.sum()) < minimum_profile_bins:
        return {**base, "profile_bin_count": int(ok.sum())}
    return {
        "eligible": True,
        "finite_fraction": finite_fraction,
        "path_span_ft": path_span,
        "relative_bins_ft": relative_bins[ok],
        "profile_gr": profile[ok],
        "profile_bin_count": int(ok.sum()),
    }


def exp226_window_score_components(
    profile_gr: np.ndarray,
    relative_bins_ft: np.ndarray,
    candidate_center_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    gr_sigma: float,
    correlation_weight: float,
    mse_weight: float,
    level_weight: float,
) -> dict[str, np.ndarray]:
    profile = np.asarray(profile_gr, dtype=np.float64)
    relative_bins = np.asarray(relative_bins_ft, dtype=np.float64)
    centers = np.asarray(candidate_center_tvt, dtype=np.float64)
    if profile.ndim != 1 or relative_bins.ndim != 1 or len(profile) != len(relative_bins):
        raise ValueError("profile values and relative bins must be aligned")
    correlation = np.full(len(centers), np.nan, dtype=np.float64)
    mse = np.full(len(centers), np.nan, dtype=np.float64)
    level = np.full(len(centers), np.nan, dtype=np.float64)
    raw_score = np.full(len(centers), np.nan, dtype=np.float64)
    profile_std = float(np.std(profile))
    profile_z = (profile - float(np.mean(profile))) / (profile_std + 1e-9)
    typewell_min = float(np.min(typewell_tvt))
    typewell_max = float(np.max(typewell_tvt))
    for slot, center in enumerate(centers):
        target_tvt = center + relative_bins
        if target_tvt.min() < typewell_min or target_tvt.max() > typewell_max:
            continue
        expected = np.interp(target_tvt, typewell_tvt, typewell_gr)
        if not np.isfinite(expected).all():
            continue
        expected_std = float(np.std(expected))
        expected_z = (expected - float(np.mean(expected))) / (expected_std + 1e-9)
        corr = float(np.mean(profile_z * expected_z))
        mse_value = float(np.mean((profile - expected) ** 2))
        level_value = float((np.mean(profile) - np.mean(expected)) ** 2)
        correlation[slot] = corr
        mse[slot] = mse_value
        level[slot] = level_value
        raw_score[slot] = (
            correlation_weight * np.arctanh(np.clip(corr, -0.95, 0.95))
            - mse_weight * mse_value / (2.0 * gr_sigma**2)
            - level_weight * level_value / (2.0 * gr_sigma**2 / 8.0)
        )
    return {
        "correlation": correlation,
        "mse": mse,
        "level": level,
        "raw_score": raw_score,
    }


def normalize_window_scores(
    raw_scores: np.ndarray,
    shifts_ft: np.ndarray,
    *,
    state_std_floor: float,
    stride_rows: int,
    window_rows: int,
    lambda_clip: tuple[float, float],
) -> dict[str, Any]:
    scores = np.asarray(raw_scores, dtype=np.float64)
    shifts = np.asarray(shifts_ft, dtype=np.float64)
    if scores.ndim != 1 or shifts.ndim != 1 or len(scores) != len(shifts):
        raise ValueError("raw scores and shifts must be aligned")
    if not np.isfinite(scores).all():
        raise ValueError("eligible window scores must be finite")
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    normalized = (scores - score_mean) / max(score_std, state_std_floor)
    weights = np.exp(normalized - float(np.max(normalized)))
    weights /= float(np.sum(weights))
    posterior_mean = float(np.sum(weights * shifts))
    posterior_sd = float(
        np.sqrt(max(float(np.sum(weights * (shifts - posterior_mean) ** 2)), 0.0))
    )
    confidence = float(np.clip(1.1 - 0.12 * posterior_sd, *lambda_clip))
    overlap = float(stride_rows / window_rows)
    lambda_value = overlap * confidence
    potential = lambda_value * normalized
    return {
        "normalized_score": normalized,
        "posterior_mean_shift_ft": posterior_mean,
        "posterior_sd_ft": posterior_sd,
        "confidence": confidence,
        "overlap_normalization": overlap,
        "lambda": lambda_value,
        "potential_score": potential,
        "score_mean": score_mean,
        "score_std": score_std,
    }


def score_well_window_target_free(
    oof_safe: pd.DataFrame,
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forbidden = set(
        str(value)
        for value in get_nested(config, "data.exp226_oof.forbidden_score_columns")
    )
    leaked = sorted(forbidden.intersection(oof_safe.columns))
    if leaked:
        raise ValueError(f"target-free OOF input contains forbidden columns: {leaked}")
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free horizontal input contains TVT")
    required = {"well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"}
    if not required.issubset(oof_safe.columns):
        raise ValueError(f"safe OOF missing {sorted(required - set(oof_safe.columns))}")
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if oof.empty or oof["well_id"].nunique() != 1 or oof["fold"].nunique() != 1:
        raise ValueError("window scorer requires one non-empty well and fold")
    suffix_offset = oof["suffix_offset"].to_numpy(np.int64)
    row_idx = oof["row_idx"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(oof), dtype=np.int64)):
        raise ValueError("suffix_offset must be contiguous from zero")
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal_without_truth):
        raise ValueError("exp226 row indices fall outside the horizontal frame")
    if horizontal_without_truth.iloc[row_idx]["TVT_input"].notna().any():
        raise ValueError("exp226 OOF must align only to unknown-suffix rows")

    prepared = prepare_gr_inputs(horizontal_without_truth, typewell, config)
    spec = get_nested(config, "model.window_potential") or {}
    shifts = np.asarray(spec["shifts_ft"], dtype=np.float64)
    window_rows = int(spec["window_rows"])
    stride_rows = int(spec["stride_rows"])
    half = window_rows // 2
    centers = list(range(half, len(oof) - half, stride_rows))
    weights = spec["score_weights"]
    geop = oof["tvt_geop"].to_numpy(np.float64)
    calibrated_gr = prepared["lateral_gr_calibrated"][row_idx]
    md = pd.to_numeric(horizontal_without_truth["MD"], errors="raise").to_numpy(np.float64)
    known_positions = np.flatnonzero(horizontal_without_truth["TVT_input"].notna().to_numpy())
    if not len(known_positions):
        raise ValueError("well has no known TVT_input prefix")
    last_known = int(known_positions[-1])
    md_since = md[row_idx] - md[last_known]
    well = str(oof["well_id"].iloc[0])
    fold = int(oof["fold"].iloc[0])
    rows: list[dict[str, Any]] = []
    eligible_windows = 0

    for window_id, center in enumerate(centers):
        start = center - half
        end = start + window_rows
        profile = build_window_profile(
            calibrated_gr[start:end],
            geop[start:end],
            grid_step_ft=float(spec["profile_grid_step_ft"]),
            minimum_finite_fraction=float(spec["minimum_finite_gr_fraction"]),
            minimum_profile_bins=int(spec["minimum_profile_bins"]),
            minimum_path_span_ft=float(spec["minimum_relative_path_span_ft"]),
        )
        profile_payload = pd.DataFrame(
            {
                "relative_tvt_ft": profile["relative_bins_ft"],
                "profile_gr": profile["profile_gr"],
            }
        )
        profile_sha = dataframe_content_sha(profile_payload)
        candidate_centers = geop[center] + shifts
        components = {
            "correlation": np.full(len(shifts), np.nan),
            "mse": np.full(len(shifts), np.nan),
            "level": np.full(len(shifts), np.nan),
            "raw_score": np.full(len(shifts), np.nan),
        }
        eligible = bool(profile["eligible"])
        normalized = np.zeros(len(shifts), dtype=np.float64)
        potential = np.zeros(len(shifts), dtype=np.float64)
        posterior_mean = 0.0
        posterior_sd = 0.0
        confidence = 0.0
        lambda_value = 0.0
        score_mean = 0.0
        score_std = 0.0
        if eligible:
            components = exp226_window_score_components(
                profile["profile_gr"],
                profile["relative_bins_ft"],
                candidate_centers,
                prepared["typewell_tvt"],
                prepared["typewell_gr"],
                gr_sigma=float(prepared["gr_sigma"]),
                correlation_weight=float(weights["correlation"]),
                mse_weight=float(weights["mse"]),
                level_weight=float(weights["level"]),
            )
            eligible = bool(np.isfinite(components["raw_score"]).all())
        if eligible:
            normalized_result = normalize_window_scores(
                components["raw_score"],
                shifts,
                state_std_floor=float(spec["state_std_floor"]),
                stride_rows=stride_rows,
                window_rows=window_rows,
                lambda_clip=tuple(float(value) for value in spec["lambda_clip"]),
            )
            normalized = normalized_result["normalized_score"]
            potential = normalized_result["potential_score"]
            posterior_mean = float(normalized_result["posterior_mean_shift_ft"])
            posterior_sd = float(normalized_result["posterior_sd_ft"])
            confidence = float(normalized_result["confidence"])
            lambda_value = float(normalized_result["lambda"])
            score_mean = float(normalized_result["score_mean"])
            score_std = float(normalized_result["score_std"])
            eligible_windows += 1
        real_ranks = rank_descending(potential)
        permutation = stable_score_permutation(well, window_id, profile_sha, len(shifts))
        shuffled = potential[permutation]
        shuffled_ranks = rank_descending(shuffled)
        control_block = int(suffix_offset[center] // int(
            get_nested(config, "data.exp280_gaussian_control.block_rows")
        ))
        for slot, shift in enumerate(shifts):
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "window_id": window_id,
                    "window_start_suffix_offset": int(suffix_offset[start]),
                    "window_end_suffix_offset": int(suffix_offset[end - 1]),
                    "window_center_suffix_offset": int(suffix_offset[center]),
                    "window_start_row_idx": int(row_idx[start]),
                    "window_end_row_idx": int(row_idx[end - 1]),
                    "window_center_row_idx": int(row_idx[center]),
                    "window_row_count": window_rows,
                    "md_since_mid_ft": float(md_since[center]),
                    "control_block_id": control_block,
                    "eligible_window": eligible,
                    "finite_gr_fraction": float(profile["finite_fraction"]),
                    "relative_path_span_ft": float(profile["path_span_ft"]),
                    "profile_bin_count": int(profile["profile_bin_count"]),
                    "profile_content_sha256": profile_sha,
                    "gr_sigma": float(prepared["gr_sigma"]),
                    "shift_slot": int(slot),
                    "shift_ft": float(shift),
                    "candidate_center_tvt": float(candidate_centers[slot]),
                    "correlation": float(components["correlation"][slot])
                    if np.isfinite(components["correlation"][slot])
                    else 0.0,
                    "mse": float(components["mse"][slot])
                    if np.isfinite(components["mse"][slot])
                    else 0.0,
                    "level": float(components["level"][slot])
                    if np.isfinite(components["level"][slot])
                    else 0.0,
                    "raw_score": float(components["raw_score"][slot])
                    if np.isfinite(components["raw_score"][slot])
                    else 0.0,
                    "score_state_mean": score_mean,
                    "score_state_std": score_std,
                    "normalized_score": float(normalized[slot]),
                    "posterior_mean_shift_ft": posterior_mean,
                    "posterior_sd_ft": posterior_sd,
                    "window_confidence": confidence,
                    "window_lambda": lambda_value,
                    "potential_score": float(potential[slot]),
                    "potential_rank": int(real_ranks[slot]),
                    "shuffled_potential_score": float(shuffled[slot]),
                    "shuffled_potential_rank": int(shuffled_ranks[slot]),
                    "shuffle_source_slot": int(permutation[slot]),
                }
            )
    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores = scores.sort_values(
            ["well_id", "window_id", "shift_slot"], kind="mergesort"
        ).reset_index(drop=True)
    manifest = {
        "well_id": well,
        "fold": fold,
        "horizontal_rows": len(horizontal_without_truth),
        "evaluation_rows": len(oof),
        "candidate_windows": len(centers),
        "eligible_windows": eligible_windows,
        "eligible_window_fraction": (
            float(eligible_windows / len(centers)) if centers else 0.0
        ),
        "known_rows": int(prepared["known_rows"]),
        "last_known_row_idx": last_known,
        "gr_sigma": float(prepared["gr_sigma"]),
        "affine_slope": float(prepared["affine_slope"]),
        "affine_intercept": float(prepared["affine_intercept"]),
        "known_residual_std_unclipped": float(
            prepared["known_residual_std_unclipped"]
        ),
    }
    return scores, manifest


# %% [markdown]
# ## 5. Saved exp280 control alignment and target-free freeze

# %%
def align_saved_control_to_windows(
    window_scores: pd.DataFrame,
    gaussian_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if window_scores.empty:
        raise ValueError("window scoring produced no candidate windows")
    control = gaussian_scores[
        [
            "well_id",
            "fold",
            "block_id",
            "shift_slot",
            "shift_ft",
            "likelihood_mean",
            "likelihood_rank",
        ]
    ].rename(
        columns={
            "fold": "control_fold",
            "block_id": "control_block_id",
            "shift_ft": "control_shift_ft",
            "likelihood_mean": "control_likelihood_mean",
            "likelihood_rank": "control_likelihood_rank",
        }
    )
    bundle = window_scores.merge(
        control,
        on=["well_id", "control_block_id", "shift_slot"],
        how="left",
        validate="many_to_one",
    )
    if len(bundle) != len(window_scores):
        raise ValueError("saved-control alignment changed target-free row count")
    required = [
        "control_fold",
        "control_shift_ft",
        "control_likelihood_mean",
        "control_likelihood_rank",
    ]
    if bundle[required].isna().any().any():
        raise ValueError("a window center did not resolve to an exp280 control block")
    if not np.array_equal(
        bundle["fold"].to_numpy(np.int64),
        bundle["control_fold"].to_numpy(np.int64),
    ):
        raise ValueError("window/control fold identities differ")
    if not np.array_equal(
        bundle["shift_ft"].to_numpy(np.float64),
        bundle["control_shift_ft"].to_numpy(np.float64),
    ):
        raise ValueError("window/control shift identities differ")
    eligible = bundle["eligible_window"].astype(bool)
    score_columns = [
        "potential_score",
        "shuffled_potential_score",
        "control_likelihood_mean",
    ]
    finite_coverage = float(
        np.isfinite(bundle.loc[eligible, score_columns].to_numpy(np.float64)).mean()
    ) if bool(eligible.any()) else 0.0
    parity_rows = []
    for _, part in bundle.groupby(["well_id", "control_block_id"], sort=False):
        first = part.drop_duplicates("shift_slot").sort_values("shift_slot")
        if len(first) != len(EXPECTED_SHIFTS):
            raise ValueError("aligned control block does not contain 13 unique shifts")
        parity_rows.append(
            np.array_equal(
                first["control_likelihood_rank"].to_numpy(np.int64),
                rank_descending(first["control_likelihood_mean"].to_numpy(np.float64)),
            )
        )
    technical = {
        "score_finite_coverage": finite_coverage,
        "row_identity_coverage": 1.0,
        "saved_control_rank_parity": float(np.mean(parity_rows)) if parity_rows else 0.0,
        "aligned_control_blocks": int(
            bundle[["well_id", "control_block_id"]].drop_duplicates().shape[0]
        ),
    }
    return bundle.sort_values(
        ["well_id", "window_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True), technical


# %% [markdown]
# ## 6. Truth-late window rank readout

# %%
def family_readout(
    score_rows: pd.DataFrame,
    candidate_rmse: np.ndarray,
    nearest_slot: int,
    *,
    score_column: str,
    rank_column: str,
    prefix: str,
) -> dict[str, Any]:
    ranks = score_rows[rank_column].to_numpy(np.int64)
    scores = score_rows[score_column].to_numpy(np.float64)
    top1_slot = int(np.argmin(ranks))
    nearest_rank = int(ranks[nearest_slot])
    ordered = np.sort(scores)[::-1]
    other = np.delete(scores, nearest_slot)
    return {
        f"{prefix}_nearest_shift_rank": nearest_rank,
        f"{prefix}_top1_hit": bool(nearest_rank == 1),
        f"{prefix}_top3_hit": bool(nearest_rank <= 3),
        f"{prefix}_mrr": float(1.0 / nearest_rank),
        f"{prefix}_top1_shift_ft": float(EXPECTED_SHIFTS[top1_slot]),
        f"{prefix}_top1_regret_rmse": float(
            candidate_rmse[top1_slot] - candidate_rmse[nearest_slot]
        ),
        f"{prefix}_top1_margin": float(ordered[0] - ordered[1]),
        f"{prefix}_truth_candidate_margin": float(
            scores[nearest_slot] - float(np.max(other))
        ),
    }


def build_truth_readout(
    target_free_bundle: pd.DataFrame,
    oof_safe: pd.DataFrame,
    truth: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if len(truth) != len(oof_safe):
        raise ValueError("truth and safe OOF row counts differ")
    merged = oof_safe.merge(truth, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    if len(merged) != len(oof_safe) or merged["tvt_true"].isna().any():
        raise ValueError("truth attachment failed row identity")
    merged = merged.sort_values(["well_id", "suffix_offset"], kind="mergesort")
    maximum_quantization_error = float(
        get_nested(config, "audit.coverage.maximum_quantization_error_ft")
    )
    readout_rows: list[dict[str, Any]] = []
    eligible_bundle = target_free_bundle.loc[
        target_free_bundle["eligible_window"].astype(bool)
    ]
    for (well, window_id), score_rows in eligible_bundle.groupby(
        ["well_id", "window_id"], sort=True
    ):
        score_rows = score_rows.sort_values("shift_slot", kind="mergesort")
        if len(score_rows) != len(EXPECTED_SHIFTS):
            raise ValueError(f"eligible window {well}/{window_id} lacks 13 shifts")
        start = int(score_rows["window_start_suffix_offset"].iloc[0])
        end = int(score_rows["window_end_suffix_offset"].iloc[0])
        well_rows = merged.loc[merged["well_id"].astype(str) == str(well)].sort_values(
            "suffix_offset", kind="mergesort"
        )
        window = well_rows.loc[well_rows["suffix_offset"].between(start, end)]
        expected_count = int(score_rows["window_row_count"].iloc[0])
        if len(window) != expected_count:
            raise ValueError(f"truth window identity mismatch for {well}/{window_id}")
        true_tvt = window["tvt_true"].to_numpy(np.float64)
        geop = window["tvt_geop"].to_numpy(np.float64)
        errors = geop[:, None] + EXPECTED_SHIFTS[None, :] - true_tvt[:, None]
        candidate_rmse = np.sqrt(np.mean(errors**2, axis=0))
        nearest_slot = int(np.argmin(candidate_rmse))
        continuous_shift = float(np.mean(true_tvt - geop))
        meta = score_rows.iloc[0]
        row = {
            "well_id": str(well),
            "fold": int(meta["fold"]),
            "window_id": int(window_id),
            "window_start_suffix_offset": start,
            "window_end_suffix_offset": end,
            "window_center_suffix_offset": int(meta["window_center_suffix_offset"]),
            "window_center_row_idx": int(meta["window_center_row_idx"]),
            "window_row_count": expected_count,
            "md_since_mid_ft": float(meta["md_since_mid_ft"]),
            "control_block_id": int(meta["control_block_id"]),
            "finite_gr_fraction": float(meta["finite_gr_fraction"]),
            "relative_path_span_ft": float(meta["relative_path_span_ft"]),
            "profile_bin_count": int(meta["profile_bin_count"]),
            "profile_content_sha256": str(meta["profile_content_sha256"]),
            "posterior_mean_shift_ft": float(meta["posterior_mean_shift_ft"]),
            "posterior_sd_ft": float(meta["posterior_sd_ft"]),
            "window_lambda": float(meta["window_lambda"]),
            "continuous_optimal_shift_ft": continuous_shift,
            "nearest_shift_slot": nearest_slot,
            "nearest_shift_ft": float(EXPECTED_SHIFTS[nearest_slot]),
            "nearest_shift_quantization_error_ft": float(
                abs(EXPECTED_SHIFTS[nearest_slot] - continuous_shift)
            ),
            "quantization_covered": bool(
                abs(EXPECTED_SHIFTS[nearest_slot] - continuous_shift)
                <= maximum_quantization_error
            ),
            "base_rmse": float(np.sqrt(np.mean((geop - true_tvt) ** 2))),
            "nearest_shift_rmse": float(candidate_rmse[nearest_slot]),
        }
        row.update(
            family_readout(
                score_rows,
                candidate_rmse,
                nearest_slot,
                score_column="potential_score",
                rank_column="potential_rank",
                prefix="window",
            )
        )
        row.update(
            family_readout(
                score_rows,
                candidate_rmse,
                nearest_slot,
                score_column="shuffled_potential_score",
                rank_column="shuffled_potential_rank",
                prefix="shuffle",
            )
        )
        row.update(
            family_readout(
                score_rows,
                candidate_rmse,
                nearest_slot,
                score_column="control_likelihood_mean",
                rank_column="control_likelihood_rank",
                prefix="control",
            )
        )
        readout_rows.append(row)
    if not readout_rows:
        raise ValueError("no eligible windows survived target-free scoring")
    return pd.DataFrame(readout_rows).sort_values(
        ["well_id", "window_id"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 7. Scope metrics and frozen Stage 0 gate

# %%
def readout_metric_row(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"scope {scope} selected zero windows")
    row: dict[str, Any] = {
        "scope": scope,
        "windows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "mean_posterior_sd_ft": float(frame["posterior_sd_ft"].mean()),
        "mean_window_lambda": float(frame["window_lambda"].mean()),
        "quantization_coverage": float(frame["quantization_covered"].mean()),
    }
    for family in ("window", "control", "shuffle"):
        row.update(
            {
                f"{family}_top1_rate": float(frame[f"{family}_top1_hit"].mean()),
                f"{family}_top3_rate": float(frame[f"{family}_top3_hit"].mean()),
                f"{family}_mrr": float(frame[f"{family}_mrr"].mean()),
                f"{family}_mean_rank": float(
                    frame[f"{family}_nearest_shift_rank"].mean()
                ),
                f"{family}_top1_regret_rmse_mean": float(
                    frame[f"{family}_top1_regret_rmse"].mean()
                ),
                f"{family}_top1_margin_mean": float(
                    frame[f"{family}_top1_margin"].mean()
                ),
            }
        )
    row.update(
        {
            "window_minus_control_top1_rate": (
                row["window_top1_rate"] - row["control_top1_rate"]
            ),
            "window_minus_control_top3_rate": (
                row["window_top3_rate"] - row["control_top3_rate"]
            ),
            "window_minus_control_mrr": row["window_mrr"] - row["control_mrr"],
            "window_minus_shuffle_top3_rate": (
                row["window_top3_rate"] - row["shuffle_top3_rate"]
            ),
            "window_minus_shuffle_mrr": row["window_mrr"] - row["shuffle_mrr"],
        }
    )
    return row


def build_scope_metrics(
    readout: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_rows = [readout_metric_row(readout, scope="overall")]
    long_limit = float(get_nested(config, "audit.scopes.long_tail_min_md_since_ft"))
    long_tail = readout.loc[readout["md_since_mid_ft"] >= long_limit]
    if long_tail.empty:
        raise ValueError("long_tail_1000_plus selected zero eligible windows")
    scope_rows.append(readout_metric_row(long_tail, scope="long_tail_1000_plus"))
    if hidden_assignments.empty:
        raise ValueError("hidden-like assignments are required by the frozen gate")
    roles = get_nested(config, "data.hidden_like.role_columns") or {}
    indexed = hidden_assignments.set_index("well_id")
    for scope_name, role_column in roles.items():
        valid_wells = set(
            indexed.index[indexed[str(role_column)].astype(str) == "valid"].astype(str)
        )
        part = readout.loc[readout["well_id"].astype(str).isin(valid_wells)]
        if part.empty:
            raise ValueError(f"{scope_name} selected zero eligible windows")
        scope_rows.append(readout_metric_row(part, scope=str(scope_name)))
    fold_rows = []
    for fold, part in readout.groupby("fold", sort=True):
        row = readout_metric_row(part, scope=f"fold_{int(fold)}")
        row["fold"] = int(fold)
        fold_rows.append(row)
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def build_by_well_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, part in readout.groupby("well_id", sort=True):
        row = readout_metric_row(part, scope=str(well))
        row["well_id"] = str(well)
        row["fold"] = int(part["fold"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_guard(
    technical_control: dict[str, Any],
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    eligible_window_fraction: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.stage_0_pass_requires_all") or {}
    scopes = scope_metrics.set_index("scope")
    required_scopes = [
        "overall",
        "long_tail_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    missing_scopes = sorted(set(required_scopes).difference(scopes.index))
    if missing_scopes:
        raise ValueError(f"required scopes missing: {missing_scopes}")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    actual_folds = sorted(int(value) for value in fold_metrics["fold"].unique())
    pooled = scopes.loc["overall"]
    improved_mrr = int((fold_metrics["window_minus_control_mrr"] > 0.0).sum())
    improved_top3 = int((fold_metrics["window_minus_control_top3_rate"] > 0.0).sum())
    real_above_shuffle = (
        (fold_metrics["window_minus_shuffle_mrr"] > 0.0)
        & (fold_metrics["window_minus_shuffle_top3_rate"] > 0.0)
    )
    direction_checks = {}
    for scope in required_scopes[1:]:
        direction_checks[scope] = bool(
            float(scopes.loc[scope, "window_minus_control_mrr"]) > 0.0
            and float(scopes.loc[scope, "window_minus_control_top3_rate"]) > 0.0
        )
    checks = {
        "expected_folds": actual_folds == expected_folds,
        "score_finite_coverage": float(technical_control["score_finite_coverage"])
        >= float(guards["required_score_finite_coverage"]),
        "row_identity_coverage": float(technical_control["row_identity_coverage"])
        >= float(guards["required_row_identity_coverage"]),
        "saved_control_rank_parity": float(technical_control["saved_control_rank_parity"])
        >= float(guards["required_control_rank_parity"]),
        "pooled_mrr_gain": float(pooled["window_minus_control_mrr"])
        >= float(guards["minimum_pooled_mrr_gain"]),
        "pooled_top3_gain": float(pooled["window_minus_control_top3_rate"])
        >= float(guards["minimum_pooled_top3_gain"]),
        "mrr_improved_folds": improved_mrr >= int(guards["minimum_improved_folds_mrr"]),
        "top3_improved_folds": improved_top3
        >= int(guards["minimum_improved_folds_top3"]),
        "real_above_shuffle_all_folds": bool(real_above_shuffle.all()),
        "long_tail_1000_plus_positive_direction": direction_checks[
            "long_tail_1000_plus"
        ],
        "hidden_like_spatial_positive_direction": direction_checks[
            "hidden_like_spatial"
        ],
        "hidden_like_typewell_purged_positive_direction": direction_checks[
            "hidden_like_typewell_purged"
        ],
        "eligible_window_fraction": eligible_window_fraction
        >= float(guards["minimum_eligible_window_fraction"]),
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "checks": checks,
        "actual_folds": actual_folds,
        "technical_control": technical_control,
        "eligible_window_fraction": eligible_window_fraction,
        "improved_folds": {"mrr": improved_mrr, "top3": improved_top3},
        "folds_real_above_shuffle": int(real_above_shuffle.sum()),
        "required_scopes": required_scopes,
        "stage_1_eligible": passed,
        "decision": (
            "stage_0_passed_stage_1_requires_separate_approval"
            if passed
            else "stage_0_failed_close_without_rescue"
        ),
    }


# %% [markdown]
# ## 8. Kaggle CPU Stage 0 orchestration

# %%
def run_stage_0_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp359 Stage 0 must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is "
            "reserved for an explicitly approved local smoke run."
        )
    validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    gaussian_control, gaussian_manifests = load_exp280_gaussian_control(config)
    safe_oof, exp226_path, exp226_manifest = load_exp226_safe(config)
    raw_dir = train_data_dir(config)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    raw_wells = sorted(
        path.name.replace("__horizontal_well.csv", "")
        for path in raw_dir.glob("*__horizontal_well.csv")
    )
    if len(raw_wells) != expected_wells or set(raw_wells) != set(
        safe_oof["well_id"].unique()
    ):
        raise ValueError("raw train and exp226 well sets differ")

    score_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, well in enumerate(raw_wells, start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        horizontal_safe = load_horizontal_without_truth(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        scores, manifest = score_well_window_target_free(
            safe_oof.loc[safe_oof["well_id"] == well],
            horizontal_safe,
            typewell,
            config,
        )
        manifest.update(
            {
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
        score_parts.append(scores)
        manifest_rows.append(manifest)
        if index % 25 == 0 or index == len(raw_wells):
            print(f"window target-free scoring wells={index}/{len(raw_wells)}")

    window_scores = pd.concat(score_parts, ignore_index=True).sort_values(
        ["well_id", "window_id", "shift_slot"], kind="mergesort"
    )
    target_free_bundle, technical_control = align_saved_control_to_windows(
        window_scores,
        gaussian_control,
    )
    target_free_content_sha = dataframe_content_sha(target_free_bundle)
    if not target_free_content_sha:
        raise RuntimeError("failed to freeze target-free window/control bundle")

    artifacts = artifact_dir()
    score_contract = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0",
        "truth_attached": False,
        "window_potential": get_nested(config, "model.window_potential"),
        "control_alignment": get_nested(config, "model.stage_0.control_alignment"),
        "negative_control": get_nested(config, "model.stage_0.negative_control"),
        "saved_control_content_sha256": get_nested(
            config, "data.exp280_gaussian_control.score_content_sha256"
        ),
        "target_free_score_content_sha256": target_free_content_sha,
    }
    score_contract["scientific_contract_sha256"] = mapping_sha256(score_contract)
    score_contract_path = artifacts / f"{OUTPUT_PREFIX}_score_contract.json"
    write_json(score_contract_path, score_contract)
    score_artifact = write_csv_gzip(
        target_free_bundle,
        artifacts / f"{OUTPUT_PREFIX}_target_free_window_scores.csv.gz",
    )

    # Unknown-suffix truth is first read here, after every target-free surface,
    # eligible mask, lambda, shuffle, and saved-control mapping has a content SHA.
    truth = load_exp226_truth(
        exp226_path,
        config,
        frozen_score_content_sha256=target_free_content_sha,
    )
    readout = build_truth_readout(target_free_bundle, safe_oof, truth, config)
    hidden, hidden_manifest = load_hidden_like_assignments(config)
    scope_metrics, fold_metrics = build_scope_metrics(readout, hidden, config)
    by_well = build_by_well_metrics(readout)
    well_manifest = pd.DataFrame(manifest_rows).sort_values("well_id", kind="mergesort")
    total_windows = int(well_manifest["candidate_windows"].sum())
    eligible_windows = int(well_manifest["eligible_windows"].sum())
    eligible_fraction = float(eligible_windows / total_windows) if total_windows else 0.0
    guard = evaluate_guard(
        technical_control,
        scope_metrics,
        fold_metrics,
        eligible_window_fraction=eligible_fraction,
        config=config,
    )
    gate_path = artifacts / f"{OUTPUT_PREFIX}_gate.json"
    write_json(gate_path, guard)
    readout_artifact = write_csv_gzip(
        readout,
        artifacts / f"{OUTPUT_PREFIX}_window_readout.csv.gz",
    )

    frames = {
        "scope_metrics": scope_metrics,
        "fold_metrics": fold_metrics,
        "by_well_metrics": by_well,
        "well_manifest": well_manifest,
    }
    frame_paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = artifacts / f"{OUTPUT_PREFIX}_{name}.csv"
        frame.to_csv(path, index=False)
        frame_paths[name] = path
    input_manifest = pd.DataFrame(
        [
            *gaussian_manifests,
            exp226_manifest,
            hidden_manifest,
            {
                "name": "raw_train_well_files",
                "path": str(raw_dir),
                "rows": int(well_manifest["horizontal_rows"].sum()),
                "wells": len(well_manifest),
                "raw_sha256": dataframe_content_sha(
                    well_manifest,
                    ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
                ),
            },
        ]
    )
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    input_manifest.to_csv(input_manifest_path, index=False)
    output_paths = {
        **frame_paths,
        "input_manifest": input_manifest_path,
        "gate": gate_path,
        "score_contract": score_contract_path,
    }
    output_sha = {name: sha256_path(path) for name, path in output_paths.items()}
    overall = scope_metrics.loc[scope_metrics["scope"] == "overall"].iloc[0].to_dict()
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_0_completed_guard_passed"
            if guard["passed"]
            else "stage_0_completed_guard_failed"
        ),
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "rows": len(safe_oof),
        "wells": int(safe_oof["well_id"].nunique()),
        "candidate_windows": total_windows,
        "eligible_windows": eligible_windows,
        "eligible_window_fraction": eligible_fraction,
        "shift_candidates": len(EXPECTED_SHIFTS),
        "scientific_scores": 1,
        "saved_control_scores": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "overall": overall,
        "guard": guard,
        "truth_attachment": {
            "stage": "after_window_score_eligibility_lambda_control_and_shuffle_freeze",
            "target_free_score_content_sha256": target_free_content_sha,
        },
        "input_manifest": input_manifest.to_dict(orient="records"),
        "artifacts": {
            "score_contract": str(score_contract_path),
            "target_free_window_scores": score_artifact,
            "window_readout": readout_artifact,
            "file_sha256": output_sha,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": guard["decision"],
        "stage_1_implemented": False,
        "stage_1_run": False,
        "inference_run": False,
        "submission_created": False,
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    write_json(
        metrics_output_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "status": summary["status"],
            "route": "pf_beam",
            "stage": "stage_0",
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "metric": get_nested(config, "validation.metric"),
            "diagnostic": {
                "overall": overall,
                "guard": guard,
                "target_free_score_content_sha256": target_free_content_sha,
            },
            "notes": (
                "Stage 0 only. No HMM, model, corrected prediction, inference, "
                "or submission is produced."
            ),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup, contract preview, and guarded execution

# %%
if in_notebook_runtime():
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG, require_run_approval=False)
    CONTRACT_PREVIEW = {
        "experiment": get_nested(CONFIG, "experiment.name"),
        "route": get_nested(CONFIG, "experiment.route"),
        "parent": get_nested(CONFIG, "lineage.parent"),
        "stage": get_nested(CONFIG, "execution.active_stage"),
        "implementation_scope": get_nested(CONFIG, "implementation.scope"),
        "window_rows": get_nested(CONFIG, "model.window_potential.window_rows"),
        "stride_rows": get_nested(CONFIG, "model.window_potential.stride_rows"),
        "shift_bank_ft": get_nested(CONFIG, "model.window_potential.shifts_ft"),
        "control_alignment": get_nested(CONFIG, "model.stage_0.control_alignment"),
        "stage_0_execution": get_nested(CONFIG, "execution.run_stage_0"),
        "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
        "hmm_well_runs": get_nested(CONFIG, "execution_contract.stage_0.hmm_well_runs"),
        "model_configs": get_nested(CONFIG, "execution_contract.stage_0.model_configs"),
        "trained_folds": get_nested(CONFIG, "execution_contract.stage_0.trained_folds"),
        "boosters": get_nested(CONFIG, "execution_contract.stage_0.boosters"),
        "stage_1_implemented": get_nested(CONFIG, "implementation.stage_1_implemented"),
        "inference_enabled": get_nested(CONFIG, "inference.enabled"),
    }
    print(json.dumps(CONTRACT_PREVIEW, indent=2, sort_keys=True))

# %% [markdown]
# The compact candidate is intentionally fail-closed until canonical notebook
# adoption and Kaggle CPU execution are separately approved.

# %%
if in_notebook_runtime():
    if bool(get_nested(CONFIG, "execution.run_stage_0")):
        STAGE_0_SUMMARY = run_stage_0_experiment(CONFIG)
    else:
        STAGE_0_SUMMARY = {
            "status": "stage_0_implemented_not_run",
            "reason": "execution.run_stage_0=false; no artifacts were generated",
        }
        print(json.dumps(STAGE_0_SUMMARY, indent=2, sort_keys=True))

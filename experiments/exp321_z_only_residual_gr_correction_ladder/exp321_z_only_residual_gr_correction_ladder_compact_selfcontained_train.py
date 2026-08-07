# %% [markdown]
# # exp321 z-only residual / GR correction ladder — Stage A/B
#
# This zero-model Kaggle CPU diagnostic builds the fixed `-delta Z` path without
# suffix truth, freezes its block identities and exp280-parity GR shift scores,
# and only then attaches saved exp226 suffix truth. Stage C, inference, and
# submission remain disabled until the frozen Stage A/B gates pass and the user
# separately approves the window-GR implementation/run.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment identity
# 2. Runtime, configuration, path, and SHA helpers
# 3. Scientific contract and input preflight
# 4. Target-free Z-only path and block construction
# 5. Exp280-parity target-free GR shift scoring
# 6. Target-free freeze contract
# 7. Late-truth Stage A residual-structure readout
# 8. Late-truth Stage B separability readout and gates
# 9. Kaggle CPU orchestration and generated artifacts
# 10. Setup, contract preview, and execution

# %%
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_NAME = "exp321_z_only_residual_gr_correction_ladder"
OUTPUT_PREFIX = "exp321_z_only"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TARGET_FREE_PATH_COLUMNS = [
    "well_id",
    "fold",
    "row_idx",
    "suffix_offset",
    "md_since_ft",
    "tvt_z",
    "tvt_geop",
    "block_h128",
    "block_h256",
    "block_h512",
]


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP321_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
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
    raise FileNotFoundError(f"exp321 config not found in {[str(path) for path in candidates]}")


def runtime_artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def runtime_metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    return json_sha256(
        [{"column": str(column), "dtype": str(frame[column].dtype)} for column in frame]
    )


def frame_content_sha256(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else [str(value) for value in columns]
    subset = frame.loc[:, chosen]
    row_hash = pd.util.hash_pandas_object(subset, index=False, categorize=True).to_numpy(
        np.uint64
    )
    digest = hashlib.sha256()
    digest.update(json.dumps(chosen, separators=(",", ":")).encode())
    digest.update(frame_schema_sha256(subset).encode())
    digest.update(np.ascontiguousarray(row_hash).tobytes())
    return digest.hexdigest()


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
        ) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as text_handle:
                frame.to_csv(text_handle, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": sha256_decompressed_gzip(path),
        "schema_sha256": frame_schema_sha256(frame),
        "logical_content_sha256": frame_content_sha256(frame),
    }


def resolve_file(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw_candidate in candidates:
        candidate = Path(str(raw_candidate))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def resolve_raw_train_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
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
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def rank_descending(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranking requires one finite score per shift")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int16)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int16)
    return ranks


# %% [markdown]
# ## 3. Scientific contract and input preflight


# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_kaggle_approval: bool = False
) -> None:
    expected_shifts = [
        -80.0,
        -40.0,
        -20.0,
        -10.0,
        -5.0,
        -2.0,
        0.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
    ]
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp321 route must remain pf_beam")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("exp321 Stage A/B implementation must be enabled")
    if get_nested(config, "implementation.scope") != "stage_ab_only":
        raise ValueError("exp321 implementation scope must remain stage_ab_only")
    if bool(get_nested(config, "implementation.stage_c_implemented")):
        raise ValueError("Stage C must remain unimplemented before the Stage A/B gate")
    if float(get_nested(config, "z_only.delta_z_coefficient")) != -1.0:
        raise ValueError("exp321 fixes the Z coefficient at -1")
    for key in ("fit_slope", "fit_intercept", "fit_rate", "use_xy", "use_donor_field", "use_ancc", "use_formation", "use_u_projection"):
        if bool(get_nested(config, f"z_only.{key}")):
            raise ValueError(f"exp321 forbids z_only.{key}")
    horizons = [int(value) for value in get_nested(config, "stage_a_residual_structure.block_horizons_rows") or []]
    if horizons != [128, 256, 512]:
        raise ValueError("Stage A fixes block horizons [128, 256, 512]")
    shifts = [float(value) for value in get_nested(config, "stage_b_shift_separability.shift_bank_ft") or []]
    if shifts != expected_shifts:
        raise ValueError("Stage B fixes the exp280 13-shift bank")
    if int(get_nested(config, "stage_b_shift_separability.block_rows") or 0) != 512:
        raise ValueError("Stage B fixes non-overlapping 512-row blocks")
    emission = get_nested(config, "stage_b_shift_separability.emission") or {}
    if emission.get("kind") != "exp209_gaussian_raw_gr":
        raise ValueError("Stage B fixes the exp209 Gaussian raw-GR emission")
    if [float(value) for value in emission.get("sigma_clip", [])] != [10.0, 60.0]:
        raise ValueError("Stage B fixes sigma clip [10, 60]")
    if float(emission.get("log_likelihood_clip", 0.0)) != 600.0:
        raise ValueError("Stage B fixes log-likelihood clip 600")
    if bool(get_nested(config, "stage_c_window_gr_correction.enabled")):
        raise ValueError("Stage C must remain disabled before Stage A/B PASS")
    stage_ab = get_nested(config, "execution_contract.stage_ab") or {}
    exact_counts = {
        "active_variants": 1,
        "diagnostic_contracts": 1,
        "fold_strata": 5,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "window_decoder_well_runs": 0,
    }
    for key, expected in exact_counts.items():
        if int(stage_ab.get(key, -1)) != expected:
            raise ValueError(f"Stage A/B requires {key}={expected}")
    if bool(get_nested(config, "execution_contract.parent_control_retraining")):
        raise ValueError("exp321 forbids parent/control retraining")
    if bool(get_nested(config, "execution_contract.gpu")):
        raise ValueError("exp321 Stage A/B must use CPU")
    if bool(get_nested(config, "execution_contract.inference")) or bool(
        get_nested(config, "execution_contract.submission")
    ):
        raise ValueError("exp321 forbids inference and submission")
    if require_kaggle_approval and not bool(
        get_nested(config, "execution_contract.kaggle_push_approved")
    ):
        raise RuntimeError("exp321 Kaggle CPU Run AB is not approved")


def load_exp226_safe(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof") or {}
    path = resolve_file(str(spec["filename"]), [str(value) for value in spec.get("candidates", [])])
    actual_decompressed_sha = sha256_decompressed_gzip(path)
    if actual_decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    safe_columns = [str(value) for value in spec["pre_freeze_safe_columns"]]
    forbidden = set(str(value) for value in spec["forbidden_pre_freeze_columns"])
    if forbidden.intersection(safe_columns):
        raise ValueError("exp226 pre-freeze allowlist contains forbidden columns")
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF has duplicate row identities")
    if not np.isfinite(frame["tvt_geop"].to_numpy(np.float64)).all():
        raise ValueError("exp226 tvt_geop must be finite")
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 safe OOF row count mismatch")
    if frame["well_id"].nunique() != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("exp226 safe OOF well count mismatch")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].unique().tolist()) != expected_folds:
        raise ValueError("exp226 fold identity mismatch")
    if not bool((frame.groupby("well_id", sort=False)["fold"].nunique() == 1).all()):
        raise ValueError("each exp226 well must map to one fold")
    manifest = {
        "name": "exp226_safe_oof",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": actual_decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
        "loaded_columns": safe_columns,
    }
    return frame, path, manifest


def load_truth_after_freeze(
    exp226_path: Path,
    config: Mapping[str, Any],
    *,
    target_free_contract_sha256: str,
) -> pd.DataFrame:
    if not target_free_contract_sha256:
        raise ValueError("truth attachment requires a frozen target-free contract SHA")
    spec = get_nested(config, "data.exp226_oof") or {}
    columns = ["well_id", "row_idx", "tvt_true"]
    if "tvt_true" not in [str(value) for value in spec["post_freeze_reference_columns"]]:
        raise ValueError("exp226 post-freeze truth allowlist is missing tvt_true")
    frame = pd.read_csv(exp226_path, usecols=columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("late truth rows must be unique")
    if not np.isfinite(frame["tvt_true"].to_numpy(np.float64)).all():
        raise ValueError("late truth values must be finite")
    return frame


def load_hidden_like_assignments(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    if not bool(spec.get("enabled")):
        return pd.DataFrame(), {"name": "hidden_like", "enabled": False}
    path = resolve_file(str(spec["filename"]), [str(value) for value in spec.get("candidates", [])])
    actual_sha = sha256_file(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *[str(value) for value in spec["role_columns"].values()]}
    if not required.issubset(frame.columns):
        raise ValueError(f"hidden-like assignment columns missing {sorted(required - set(frame.columns))}")
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments require one row per well")
    return frame, {
        "name": "exp115_hidden_like_assignments",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


def load_horizontal_without_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda column: column != "TVT")
    if "TVT" in frame.columns:
        raise ValueError("target-free horizontal reader must not expose TVT")
    return frame


# %% [markdown]
# ## 4. Target-free Z-only path and block construction


# %%
def build_z_only_target_free(
    oof_safe: pd.DataFrame,
    horizontal_without_truth: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forbidden = set(
        str(value)
        for value in get_nested(config, "data.exp226_oof.forbidden_pre_freeze_columns")
    )
    leaked = sorted(forbidden.intersection(oof_safe.columns))
    if leaked:
        raise ValueError(f"target-free Z path contains forbidden columns: {leaked}")
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free Z path forbids raw suffix TVT")
    required_raw = {"MD", "Z", "GR", "TVT_input"}
    if not required_raw.issubset(horizontal_without_truth.columns):
        raise ValueError(f"raw horizontal missing {sorted(required_raw - set(horizontal_without_truth.columns))}")
    required_oof = {"well_id", "fold", "row_idx", "suffix_offset", "tvt_geop"}
    if not required_oof.issubset(oof_safe.columns):
        raise ValueError(f"safe OOF missing {sorted(required_oof - set(oof_safe.columns))}")
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if oof.empty or oof["well_id"].nunique() != 1 or oof["fold"].nunique() != 1:
        raise ValueError("Z-only builder requires one non-empty well and fold")
    tvt_input = pd.to_numeric(horizontal_without_truth["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_positions = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known_positions):
        raise ValueError("well has no finite TVT_input anchor")
    last_known = int(known_positions[-1])
    if not np.array_equal(known_positions, np.arange(last_known + 1, dtype=np.int64)):
        raise ValueError("TVT_input must be a contiguous known prefix")
    row_idx = oof["row_idx"].to_numpy(np.int64)
    expected_row_idx = np.arange(last_known + 1, len(horizontal_without_truth), dtype=np.int64)
    if not np.array_equal(row_idx, expected_row_idx):
        raise ValueError("exp226 suffix row identity must equal the raw unknown suffix")
    suffix_offset = oof["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(oof), dtype=np.int64)):
        raise ValueError("suffix_offset must be contiguous from zero")
    z = pd.to_numeric(horizontal_without_truth["Z"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(z[last_known]):
        raise ValueError("last known anchor Z must be finite")
    if not np.isfinite(z[row_idx]).all():
        raise ValueError("all suffix Z values must be finite; interpolation is forbidden")
    anchor_tvt = float(tvt_input[last_known])
    anchor_z = float(z[last_known])
    tvt_z = anchor_tvt - (z[row_idx] - anchor_z)
    if not np.isfinite(tvt_z).all():
        raise ValueError("Z-only path must be finite")
    md = pd.to_numeric(horizontal_without_truth["MD"], errors="raise").to_numpy(np.float64)
    md_since = md[row_idx] - md[last_known]
    frame = pd.DataFrame(
        {
            "well_id": str(oof["well_id"].iloc[0]),
            "fold": int(oof["fold"].iloc[0]),
            "row_idx": row_idx,
            "suffix_offset": suffix_offset,
            "md_since_ft": md_since,
            "tvt_z": tvt_z,
            "tvt_geop": oof["tvt_geop"].to_numpy(np.float64),
        }
    )
    for horizon in [int(value) for value in get_nested(config, "stage_a_residual_structure.block_horizons_rows")]:
        frame[f"block_h{horizon}"] = suffix_offset // horizon
    frame = frame[TARGET_FREE_PATH_COLUMNS]
    manifest = {
        "well_id": str(oof["well_id"].iloc[0]),
        "fold": int(oof["fold"].iloc[0]),
        "horizontal_rows": len(horizontal_without_truth),
        "known_prefix_rows": last_known + 1,
        "suffix_rows": len(frame),
        "last_known_row_idx": last_known,
        "anchor_tvt": anchor_tvt,
        "anchor_z": anchor_z,
        "finite_z_coverage": float(np.isfinite(z[row_idx]).mean()),
        "finite_tvt_z_coverage": float(np.isfinite(tvt_z).mean()),
        "row_identity_coverage": 1.0,
    }
    return frame, manifest


# %% [markdown]
# ## 5. Exp280-parity target-free GR shift scoring


# %%
def prepare_gr_inputs(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free GR preparation forbids horizontal TVT")
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
    tw = typewell[["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw[["TVT", "GR"]].to_numpy(np.float64)).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    known = horizontal_without_truth.loc[horizontal_without_truth["TVT_input"].notna()]
    if len(known) < 4:
        raise ValueError("well requires at least four known-prefix rows")
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma_low, sigma_high = [
        float(value)
        for value in get_nested(config, "stage_b_shift_separability.emission.sigma_clip")
    ]
    sigma = float(np.clip(np.nanstd(residual), sigma_low, sigma_high))
    if not np.isfinite(sigma):
        raise ValueError("known-prefix GR residual sigma must be finite")
    gr_fill = float(np.nanmean(typewell_gr))
    all_gr = (
        pd.to_numeric(horizontal_without_truth["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)
    )
    return {
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "gr_sigma": sigma,
        "all_gr_interpolated": all_gr,
        "known_rows": len(known),
        "known_residual_mean": float(np.mean(residual)),
        "known_residual_std_unclipped": float(np.std(residual)),
    }


def score_z_only_shifts_target_free(
    path_frame: pd.DataFrame,
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forbidden = {"tvt_true", "TVT", "error", "abs_error", "gr_delta"}
    leaked = sorted(forbidden.intersection(path_frame.columns))
    if leaked:
        raise ValueError(f"Stage B target-free input contains forbidden columns: {leaked}")
    prepared = prepare_gr_inputs(horizontal_without_truth, typewell, config)
    frame = path_frame.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    shifts = np.asarray(get_nested(config, "stage_b_shift_separability.shift_bank_ft"), dtype=np.float64)
    block_rows = int(get_nested(config, "stage_b_shift_separability.block_rows"))
    row_idx = frame["row_idx"].to_numpy(np.int64)
    suffix_offset = frame["suffix_offset"].to_numpy(np.int64)
    candidate_tvt = frame["tvt_z"].to_numpy(np.float64)[:, None] + shifts[None, :]
    expected_gr = np.empty_like(candidate_tvt)
    for slot in range(len(shifts)):
        expected_gr[:, slot] = np.interp(
            candidate_tvt[:, slot], prepared["typewell_tvt"], prepared["typewell_gr"]
        )
    raw_gr = prepared["all_gr_interpolated"][row_idx]
    clip_value = float(
        get_nested(config, "stage_b_shift_separability.emission.log_likelihood_clip")
    )
    zscore = (raw_gr[:, None] - expected_gr) / float(prepared["gr_sigma"])
    log_likelihood = -0.5 * np.minimum(np.square(zscore), clip_value)
    if not np.isfinite(log_likelihood).all():
        raise ValueError("Stage B target-free likelihood must be finite")
    block_id = suffix_offset // block_rows
    native = (candidate_tvt >= prepared["typewell_tvt"].min()) & (
        candidate_tvt <= prepared["typewell_tvt"].max()
    )
    extension = float(get_nested(config, "stage_b_shift_separability.typewell_extension_ft"))
    extended = (candidate_tvt >= prepared["typewell_tvt"].min() - extension) & (
        candidate_tvt <= prepared["typewell_tvt"].max() + extension
    )
    observed_gr = pd.to_numeric(horizontal_without_truth.iloc[row_idx]["GR"], errors="coerce")
    well = str(frame["well_id"].iloc[0])
    fold = int(frame["fold"].iloc[0])
    shuffle_seed = int(get_nested(config, "stage_b_shift_separability.shuffled_control.seed"))
    rows: list[dict[str, Any]] = []
    for block in np.unique(block_id):
        mask = block_id == block
        positions = np.flatnonzero(mask)
        scores = log_likelihood[mask].mean(axis=0)
        sums = log_likelihood[mask].sum(axis=0)
        ranks = rank_descending(scores)
        rng = np.random.default_rng(
            stable_seed(EXPERIMENT_NAME, shuffle_seed, well, int(block))
        )
        shuffled_scores = scores[rng.permutation(len(scores))]
        shuffled_ranks = rank_descending(shuffled_scores)
        for slot, shift in enumerate(shifts):
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "block_id": int(block),
                    "block_start_suffix_offset": int(suffix_offset[positions[0]]),
                    "block_end_suffix_offset": int(suffix_offset[positions[-1]]),
                    "block_start_row_idx": int(row_idx[positions[0]]),
                    "block_end_row_idx": int(row_idx[positions[-1]]),
                    "block_row_count": int(mask.sum()),
                    "md_since_min_ft": float(frame.loc[mask, "md_since_ft"].min()),
                    "md_since_max_ft": float(frame.loc[mask, "md_since_ft"].max()),
                    "md_since_mid_ft": float(frame.loc[mask, "md_since_ft"].mean()),
                    "observed_gr_share": float(observed_gr.iloc[positions].notna().mean()),
                    "shift_slot": int(slot),
                    "shift_ft": float(shift),
                    "likelihood_mean": float(scores[slot]),
                    "likelihood_sum": float(sums[slot]),
                    "likelihood_rank": int(ranks[slot]),
                    "shuffled_likelihood_mean": float(shuffled_scores[slot]),
                    "shuffled_likelihood_rank": int(shuffled_ranks[slot]),
                    "native_typewell_coverage": float(native[mask, slot].mean()),
                    "extended_typewell_coverage": float(extended[mask, slot].mean()),
                }
            )
    score_frame = pd.DataFrame(rows).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    manifest = {
        "blocks": int(block_id.max() + 1),
        "gr_sigma": float(prepared["gr_sigma"]),
        "known_residual_mean": float(prepared["known_residual_mean"]),
        "known_residual_std_unclipped": float(prepared["known_residual_std_unclipped"]),
        "observed_eval_gr_share": float(observed_gr.notna().mean()),
        "score_finite_coverage": float(np.isfinite(log_likelihood).mean()),
    }
    return score_frame.reset_index(drop=True), manifest


# %% [markdown]
# ## 6. Target-free freeze contract


# %%
def build_target_free_contract(
    config: Mapping[str, Any],
    path_artifact: Mapping[str, Any],
    score_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_ab_target_free_freeze",
        "truth_attached": False,
        "z_only": get_nested(config, "z_only"),
        "stage_a_block_horizons_rows": get_nested(
            config, "stage_a_residual_structure.block_horizons_rows"
        ),
        "stage_b": {
            "shift_bank_ft": get_nested(config, "stage_b_shift_separability.shift_bank_ft"),
            "block_rows": get_nested(config, "stage_b_shift_separability.block_rows"),
            "block_policy": get_nested(config, "stage_b_shift_separability.block_policy"),
            "score_aggregation": get_nested(
                config, "stage_b_shift_separability.score_aggregation"
            ),
            "tie_policy": get_nested(config, "stage_b_shift_separability.tie_policy"),
            "emission": get_nested(config, "stage_b_shift_separability.emission"),
            "shuffled_control": get_nested(
                config, "stage_b_shift_separability.shuffled_control"
            ),
        },
        "path_evidence": dict(path_artifact),
        "score_evidence": dict(score_artifact),
    }
    contract["target_free_contract_sha256"] = json_sha256(contract)
    return contract


# %% [markdown]
# ## 7. Late-truth Stage A residual-structure readout


# %%
def attach_truth_after_freeze(
    target_free_paths: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    target_free_contract_sha256: str,
) -> pd.DataFrame:
    if not target_free_contract_sha256:
        raise ValueError("late truth join requires a frozen target-free contract")
    if "tvt_true" in target_free_paths.columns:
        raise ValueError("target-free paths unexpectedly contain truth")
    joined = target_free_paths.merge(
        truth, on=["well_id", "row_idx"], how="left", validate="one_to_one"
    )
    if len(joined) != len(target_free_paths) or joined["tvt_true"].isna().any():
        raise ValueError("late truth join failed full row identity coverage")
    if not np.isfinite(joined["tvt_true"].to_numpy(np.float64)).all():
        raise ValueError("late truth join produced non-finite truth")
    return joined.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)


def _residual_block_stats(residual: np.ndarray) -> dict[str, Any]:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("block residual must be a non-empty finite vector")
    n_rows = len(values)
    x = np.arange(n_rows, dtype=np.float64)
    mean = float(values.mean())
    centered = values - mean
    direct_sse = float(np.dot(values, values))
    offset_sse = float(np.dot(centered, centered))
    affine_valid = n_rows >= 2
    slope = np.nan
    affine_sse = np.nan
    if affine_valid:
        x_centered = x - x.mean()
        x_sse = float(np.dot(x_centered, x_centered))
        slope = float(np.dot(x_centered, centered) / x_sse)
        affine_residual = centered - slope * x_centered
        affine_sse = float(max(np.dot(affine_residual, affine_residual), 0.0))
    lag1 = np.nan
    if n_rows >= 2 and np.std(values[:-1]) > 0.0 and np.std(values[1:]) > 0.0:
        lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    correction = float(np.clip(mean, -4.0, 4.0))
    cap4_sse = float(np.dot(values - correction, values - correction))
    return {
        "rows": n_rows,
        "direct_sse": direct_sse,
        "offset_sse": offset_sse,
        "affine_sse": affine_sse,
        "affine_valid": affine_valid,
        "block_mean_residual_ft": mean,
        "block_slope_ft_per_row": slope,
        "lag1_correlation": lag1,
        "cap4_oracle_correction_ft": correction,
        "cap4_oracle_sse": cap4_sse,
    }


def build_stage_a_block_readout(
    joined: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizons = [
        int(value)
        for value in get_nested(config, "stage_a_residual_structure.block_horizons_rows")
    ]
    for horizon in horizons:
        block_column = f"block_h{horizon}"
        for (well, block_id), part in joined.groupby(
            ["well_id", block_column], sort=True, observed=True
        ):
            part = part.sort_values("row_idx", kind="mergesort")
            truth = part["tvt_true"].to_numpy(np.float64)
            z_stats = _residual_block_stats(truth - part["tvt_z"].to_numpy(np.float64))
            geop_stats = _residual_block_stats(
                truth - part["tvt_geop"].to_numpy(np.float64)
            )
            row: dict[str, Any] = {
                "horizon_rows": horizon,
                "well_id": str(well),
                "fold": int(part["fold"].iloc[0]),
                "block_id": int(block_id),
                "block_start_row_idx": int(part["row_idx"].iloc[0]),
                "block_end_row_idx": int(part["row_idx"].iloc[-1]),
                "block_start_suffix_offset": int(part["suffix_offset"].iloc[0]),
                "block_end_suffix_offset": int(part["suffix_offset"].iloc[-1]),
                "md_since_min_ft": float(part["md_since_ft"].min()),
                "md_since_max_ft": float(part["md_since_ft"].max()),
            }
            row.update({f"z_{key}": value for key, value in z_stats.items()})
            row.update({f"geop_{key}": value for key, value in geop_stats.items()})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["horizon_rows", "well_id", "block_id"], kind="mergesort"
    ).reset_index(drop=True)


def _stage_a_metric_row(blocks: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    if blocks.empty:
        raise ValueError(f"Stage A scope {scope} has zero blocks")
    rows = int(blocks["z_rows"].sum())
    affine = blocks["z_affine_valid"].astype(bool)
    affine_rows = int(blocks.loc[affine, "z_rows"].sum())
    z_direct_sse = float(blocks["z_direct_sse"].sum())
    geop_direct_sse = float(blocks["geop_direct_sse"].sum())
    z_affine_sse = float(blocks.loc[affine, "z_affine_sse"].sum())
    geop_affine_sse = float(blocks.loc[affine, "geop_affine_sse"].sum())
    z_direct_eligible_sse = float(blocks.loc[affine, "z_direct_sse"].sum())
    z_affine_rmse = math.sqrt(z_affine_sse / affine_rows)
    geop_affine_rmse = math.sqrt(geop_affine_sse / affine_rows)
    cap4_rmse = math.sqrt(float(blocks["z_cap4_oracle_sse"].sum()) / rows)
    z_direct_rmse = math.sqrt(z_direct_sse / rows)
    return {
        "scope": scope,
        "horizon_rows": int(blocks["horizon_rows"].iloc[0]),
        "rows": rows,
        "wells": int(blocks["well_id"].nunique()),
        "blocks": len(blocks),
        "affine_eligible_rows": affine_rows,
        "affine_excluded_singleton_blocks": int((~affine).sum()),
        "z_direct_rmse": z_direct_rmse,
        "geop_direct_rmse": math.sqrt(geop_direct_sse / rows),
        "z_offset_quotient_rmse": math.sqrt(float(blocks["z_offset_sse"].sum()) / rows),
        "geop_offset_quotient_rmse": math.sqrt(
            float(blocks["geop_offset_sse"].sum()) / rows
        ),
        "z_affine_quotient_rmse": z_affine_rmse,
        "geop_affine_quotient_rmse": geop_affine_rmse,
        "z_vs_geop_affine_rmse_ratio": z_affine_rmse / geop_affine_rmse
        if geop_affine_rmse > 0.0
        else (1.0 if z_affine_rmse == 0.0 else np.inf),
        "z_affine_sse_explained_fraction": 1.0
        - z_affine_sse / z_direct_eligible_sse
        if z_direct_eligible_sse > 0.0
        else 0.0,
        "z_cap4_oracle_rmse": cap4_rmse,
        "z_cap4_oracle_rmse_gain_ft": z_direct_rmse - cap4_rmse,
        "z_block_mean_abs_ft": float(blocks["z_block_mean_residual_ft"].abs().mean()),
        "z_block_slope_abs_ft_per_row": float(
            blocks["z_block_slope_ft_per_row"].abs().mean()
        ),
        "z_lag1_correlation_mean": float(blocks["z_lag1_correlation"].mean()),
    }


def build_stage_a_metrics(blocks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, part in blocks.groupby("horizon_rows", sort=True):
        rows.append(_stage_a_metric_row(part, scope="overall"))
        for fold, fold_part in part.groupby("fold", sort=True):
            row = _stage_a_metric_row(fold_part, scope=f"fold_{int(fold)}")
            row["fold"] = int(fold)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["horizon_rows", "scope"], kind="mergesort"
    ).reset_index(drop=True)


def evaluate_stage_a_gate(
    metrics: pd.DataFrame,
    *,
    row_identity_coverage: float,
    well_identity_coverage: float,
    finite_prediction_coverage: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "stage_a_residual_structure.pass_requires_all") or {}
    fold_required = int(guards["minimum_folds_meeting_relative_shape_each_primary_horizon"])
    fold_counts: dict[str, int] = {}
    checks: dict[str, bool] = {}
    for horizon in (256, 512):
        fold_rows = metrics.loc[
            (metrics["horizon_rows"] == horizon) & metrics["scope"].str.startswith("fold_")
        ]
        threshold = float(
            guards[f"maximum_affine_quotient_rmse_ratio_vs_exp226_tvt_geop_h{horizon}"]
        )
        fold_counts[f"h{horizon}"] = int(
            (fold_rows["z_vs_geop_affine_rmse_ratio"] <= threshold).sum()
        )
        checks[f"h{horizon}_relative_shape_folds"] = fold_counts[f"h{horizon}"] >= fold_required
        overall = metrics.loc[
            (metrics["horizon_rows"] == horizon) & (metrics["scope"] == "overall")
        ].iloc[0]
        checks[f"h{horizon}_overall_relative_shape"] = bool(
            overall["z_vs_geop_affine_rmse_ratio"] <= threshold
        )
    h512 = metrics.loc[
        (metrics["horizon_rows"] == 512) & (metrics["scope"] == "overall")
    ].iloc[0]
    checks.update(
        {
            "h512_affine_sse_explained": bool(
                h512["z_affine_sse_explained_fraction"]
                >= float(guards["minimum_h512_affine_sse_explained_fraction"])
            ),
            "h512_cap4_oracle_headroom": bool(
                h512["z_cap4_oracle_rmse_gain_ft"]
                >= float(guards["minimum_h512_cap4_oracle_rmse_gain_ft"])
            ),
            "row_identity_coverage": row_identity_coverage
            >= float(guards["required_technical_coverage"]),
            "well_identity_coverage": well_identity_coverage
            >= float(guards["required_technical_coverage"]),
            "finite_prediction_coverage": finite_prediction_coverage
            >= float(guards["required_technical_coverage"]),
        }
    )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "folds_meeting_relative_shape": fold_counts,
        "required_folds": fold_required,
        "technical_coverage": {
            "row_identity": row_identity_coverage,
            "well_identity": well_identity_coverage,
            "finite_prediction": finite_prediction_coverage,
        },
    }


# %% [markdown]
# ## 8. Late-truth Stage B separability readout and gates


# %%
def sign_match(selected_shift: float, nearest_shift: float) -> bool:
    return bool(np.sign(float(selected_shift)) == np.sign(float(nearest_shift)))


def build_stage_b_block_readout(
    target_free_scores: pd.DataFrame,
    joined: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    shifts = np.asarray(get_nested(config, "stage_b_shift_separability.shift_bank_ft"), dtype=np.float64)
    block_rows = int(get_nested(config, "stage_b_shift_separability.block_rows"))
    maximum_quantization_error = float(
        get_nested(
            config,
            "stage_b_shift_separability.pass_requires_all.maximum_quantization_error_ft",
        )
    )
    rows: list[dict[str, Any]] = []
    frame = joined.copy()
    frame["stage_b_block_id"] = frame["suffix_offset"] // block_rows
    for (well, block_id), part in frame.groupby(
        ["well_id", "stage_b_block_id"], sort=True, observed=True
    ):
        part = part.sort_values("row_idx", kind="mergesort")
        scores = target_free_scores.loc[
            (target_free_scores["well_id"].astype(str) == str(well))
            & (target_free_scores["block_id"] == int(block_id))
        ].sort_values("shift_slot", kind="mergesort")
        if len(scores) != len(shifts) or not np.array_equal(
            scores["shift_ft"].to_numpy(np.float64), shifts
        ):
            raise ValueError(f"Stage B score bank misalignment for {well} block {block_id}")
        truth = part["tvt_true"].to_numpy(np.float64)
        base = part["tvt_z"].to_numpy(np.float64)
        errors = base[:, None] + shifts[None, :] - truth[:, None]
        candidate_rmse = np.sqrt(np.mean(np.square(errors), axis=0))
        nearest_slot = int(np.argmin(candidate_rmse))
        real_rank = int(scores["likelihood_rank"].iloc[nearest_slot])
        shuffled_rank = int(scores["shuffled_likelihood_rank"].iloc[nearest_slot])
        top1_slot = int(np.argmin(scores["likelihood_rank"].to_numpy(np.int64)))
        shuffled_top1_slot = int(
            np.argmin(scores["shuffled_likelihood_rank"].to_numpy(np.int64))
        )
        likelihood = scores["likelihood_mean"].to_numpy(np.float64)
        ordered_likelihood = np.sort(likelihood)[::-1]
        other = np.delete(likelihood, nearest_slot)
        continuous_optimal_shift = float(np.mean(truth - base))
        nearest_shift = float(shifts[nearest_slot])
        top1_shift = float(shifts[top1_slot])
        shuffled_top1_shift = float(shifts[shuffled_top1_slot])
        quantization_error = float(abs(nearest_shift - continuous_optimal_shift))
        rows.append(
            {
                "well_id": str(well),
                "fold": int(part["fold"].iloc[0]),
                "block_id": int(block_id),
                "block_start_row_idx": int(part["row_idx"].iloc[0]),
                "block_end_row_idx": int(part["row_idx"].iloc[-1]),
                "block_row_count": len(part),
                "md_since_min_ft": float(part["md_since_ft"].min()),
                "md_since_max_ft": float(part["md_since_ft"].max()),
                "md_since_mid_ft": float(part["md_since_ft"].mean()),
                "observed_gr_share": float(scores["observed_gr_share"].iloc[0]),
                "continuous_optimal_shift_ft": continuous_optimal_shift,
                "nearest_shift_ft": nearest_shift,
                "nearest_shift_slot": nearest_slot,
                "nearest_shift_rank": real_rank,
                "nearest_shift_shuffled_rank": shuffled_rank,
                "top1_hit": bool(real_rank == 1),
                "top3_hit": bool(real_rank <= 3),
                "mrr": float(1.0 / real_rank),
                "shuffled_top1_hit": bool(shuffled_rank == 1),
                "shuffled_top3_hit": bool(shuffled_rank <= 3),
                "shuffled_mrr": float(1.0 / shuffled_rank),
                "top1_shift_ft": top1_shift,
                "shuffled_top1_shift_ft": shuffled_top1_shift,
                "sign_match": sign_match(top1_shift, nearest_shift),
                "shuffled_sign_match": sign_match(shuffled_top1_shift, nearest_shift),
                "likelihood_top1_margin": float(ordered_likelihood[0] - ordered_likelihood[1]),
                "truth_candidate_margin": float(likelihood[nearest_slot] - np.max(other)),
                "base_rmse": float(np.sqrt(np.mean(np.square(base - truth)))),
                "nearest_shift_rmse": float(candidate_rmse[nearest_slot]),
                "top1_shift_rmse": float(candidate_rmse[top1_slot]),
                "top1_regret_rmse": float(
                    candidate_rmse[top1_slot] - candidate_rmse[nearest_slot]
                ),
                "oracle_shift_gain_rmse": float(
                    np.sqrt(np.mean(np.square(base - truth))) - candidate_rmse[nearest_slot]
                ),
                "bank_range_covered": bool(
                    shifts.min() <= continuous_optimal_shift <= shifts.max()
                ),
                "nearest_shift_quantization_error_ft": quantization_error,
                "quantization_covered": bool(
                    quantization_error <= maximum_quantization_error
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["well_id", "block_id"], kind="mergesort"
    ).reset_index(drop=True)


def stage_b_metric_row(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"Stage B scope {scope} has zero blocks")
    return {
        "scope": scope,
        "blocks": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "top1": float(frame["top1_hit"].mean()),
        "top3": float(frame["top3_hit"].mean()),
        "mrr": float(frame["mrr"].mean()),
        "sign": float(frame["sign_match"].mean()),
        "shuffled_top1": float(frame["shuffled_top1_hit"].mean()),
        "shuffled_top3": float(frame["shuffled_top3_hit"].mean()),
        "shuffled_mrr": float(frame["shuffled_mrr"].mean()),
        "shuffled_sign": float(frame["shuffled_sign_match"].mean()),
        "top1_lift_vs_shuffled": float(
            frame["top1_hit"].mean() - frame["shuffled_top1_hit"].mean()
        ),
        "top3_lift_vs_shuffled": float(
            frame["top3_hit"].mean() - frame["shuffled_top3_hit"].mean()
        ),
        "mrr_lift_vs_shuffled": float(frame["mrr"].mean() - frame["shuffled_mrr"].mean()),
        "sign_lift_vs_shuffled": float(
            frame["sign_match"].mean() - frame["shuffled_sign_match"].mean()
        ),
        "bank_range_coverage": float(frame["bank_range_covered"].mean()),
        "quantization_coverage": float(frame["quantization_covered"].mean()),
        "maximum_quantization_error_ft": float(
            frame["nearest_shift_quantization_error_ft"].max()
        ),
        "oracle_shift_gain_rmse_mean": float(frame["oracle_shift_gain_rmse"].mean()),
        "top1_regret_rmse_mean": float(frame["top1_regret_rmse"].mean()),
        "likelihood_top1_margin_mean": float(frame["likelihood_top1_margin"].mean()),
    }


def build_stage_b_metrics(
    readout: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope_rows = [stage_b_metric_row(readout, scope="overall")]
    long_min = float(get_nested(config, "validation.scopes.long_tail.minimum_md_since_ft"))
    near_max = float(get_nested(config, "validation.scopes.near.maximum_md_since_ft"))
    predefined = {
        "near_0_250": readout["md_since_mid_ft"] <= near_max,
        "long_tail_1000_plus": readout["md_since_mid_ft"] >= long_min,
    }
    for scope, mask in predefined.items():
        if bool(mask.any()):
            scope_rows.append(stage_b_metric_row(readout.loc[mask], scope=scope))
    if not hidden_assignments.empty:
        roles = get_nested(config, "data.hidden_like.role_columns") or {}
        indexed = hidden_assignments.set_index("well_id")
        for scope, role_column in roles.items():
            valid_wells = set(
                indexed.index[indexed[str(role_column)].astype(str) == "valid"].astype(str)
            )
            scope_rows.append(
                stage_b_metric_row(
                    readout.loc[readout["well_id"].astype(str).isin(valid_wells)],
                    scope=str(scope),
                )
            )
    fold_rows: list[dict[str, Any]] = []
    for fold, part in readout.groupby("fold", sort=True):
        row = stage_b_metric_row(part, scope=f"fold_{int(fold)}")
        row["fold"] = int(fold)
        fold_rows.append(row)
    shift_rows: list[dict[str, Any]] = []
    shifts = [float(value) for value in get_nested(config, "stage_b_shift_separability.shift_bank_ft")]
    for shift in shifts:
        nearest = readout.loc[np.isclose(readout["nearest_shift_ft"], shift)]
        selected = readout.loc[np.isclose(readout["top1_shift_ft"], shift)]
        shift_rows.append(
            {
                "shift_ft": shift,
                "truth_nearest_blocks": len(nearest),
                "truth_nearest_share": float(len(nearest) / len(readout)),
                "likelihood_top1_blocks": len(selected),
                "likelihood_top1_share": float(len(selected) / len(readout)),
                "top1_when_truth_nearest": float(nearest["top1_hit"].mean())
                if len(nearest)
                else np.nan,
                "top3_when_truth_nearest": float(nearest["top3_hit"].mean())
                if len(nearest)
                else np.nan,
            }
        )
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows), pd.DataFrame(shift_rows)


def evaluate_stage_b_gate(
    target_free_scores: pd.DataFrame,
    readout: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "stage_b_shift_separability.pass_requires_all") or {}
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    actual_folds = sorted(int(value) for value in fold_metrics["fold"].unique())
    metric_pairs = {
        "top1": ("top1", "shuffled_top1"),
        "top3": ("top3", "shuffled_top3"),
        "mrr": ("mrr", "shuffled_mrr"),
        "sign": ("sign", "shuffled_sign"),
    }
    fold_counts = {
        name: int((fold_metrics[real] > fold_metrics[shuffled]).sum())
        for name, (real, shuffled) in metric_pairs.items()
    }
    required_fold_count = int(guards["minimum_folds_real_above_shuffled_each_metric"])
    overall = scope_metrics.loc[scope_metrics["scope"] == "overall"].iloc[0]
    references = get_nested(config, "stage_b_shift_separability.exp280_pooled_reference") or {}
    checks: dict[str, bool] = {
        "expected_folds": actual_folds == expected_folds,
        "finite_score_coverage": bool(
            np.isfinite(
                target_free_scores[["likelihood_mean", "shuffled_likelihood_mean"]].to_numpy(np.float64)
            ).mean()
            >= float(guards["required_finite_score_coverage"])
        ),
        "row_identity_coverage": int(readout["block_row_count"].sum())
        == int(get_nested(config, "validation.expected_rows")),
        "bank_range_coverage": bool(
            overall["bank_range_coverage"] >= float(guards["minimum_bank_range_coverage"])
        ),
        "maximum_quantization_error": bool(
            overall["maximum_quantization_error_ft"]
            <= float(guards["maximum_quantization_error_ft"])
        ),
    }
    for name, count in fold_counts.items():
        checks[f"{name}_real_above_shuffle_all_folds"] = count >= required_fold_count
        checks[f"{name}_strictly_above_exp280_pooled"] = bool(
            overall[name] > float(references[name])
        )
    required_scopes = [str(value) for value in guards["required_positive_real_minus_shuffled_scopes"]]
    scope_checks: dict[str, dict[str, bool]] = {}
    for scope in required_scopes:
        matches = scope_metrics.loc[scope_metrics["scope"] == scope]
        if len(matches) != 1:
            scope_checks[scope] = {name: False for name in metric_pairs}
        else:
            row = matches.iloc[0]
            scope_checks[scope] = {
                name: bool(row[real] > row[shuffled])
                for name, (real, shuffled) in metric_pairs.items()
            }
        checks[f"{scope}_all_four_positive"] = bool(all(scope_checks[scope].values()))
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "actual_folds": actual_folds,
        "folds_real_above_shuffled": fold_counts,
        "required_folds": required_fold_count,
        "scope_checks": scope_checks,
        "overall": overall.to_dict(),
    }


# %% [markdown]
# ## 9. Kaggle CPU orchestration and generated artifacts


# %%
def run_stage_ab(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp321 Stage A/B must run on Kaggle; local execution requires explicit approval"
        )
    validate_scientific_contract(config, require_kaggle_approval=True)
    started = time.time()
    safe_oof, exp226_path, exp226_manifest = load_exp226_safe(config)
    raw_dir = resolve_raw_train_dir(config)
    raw_wells = sorted(
        path.name.replace("__horizontal_well.csv", "")
        for path in raw_dir.glob("*__horizontal_well.csv")
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(raw_wells) != expected_wells or set(raw_wells) != set(safe_oof["well_id"].unique()):
        raise ValueError("raw-train and exp226 well identities do not match")
    path_parts: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    well_manifest_rows: list[dict[str, Any]] = []
    for index, well in enumerate(raw_wells, start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            raise FileNotFoundError(typewell_path)
        horizontal_safe = load_horizontal_without_truth(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        z_path, z_manifest = build_z_only_target_free(
            safe_oof.loc[safe_oof["well_id"] == well], horizontal_safe, config
        )
        scores, score_manifest = score_z_only_shifts_target_free(
            z_path, horizontal_safe, typewell, config
        )
        z_manifest.update(score_manifest)
        z_manifest.update(
            {
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_file(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_file(typewell_path),
            }
        )
        path_parts.append(z_path)
        score_parts.append(scores)
        well_manifest_rows.append(z_manifest)
        if index % 25 == 0 or index == len(raw_wells):
            print(f"target-free Stage A/B wells={index}/{len(raw_wells)}")
    target_free_paths = pd.concat(path_parts, ignore_index=True).sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    target_free_scores = pd.concat(score_parts, ignore_index=True).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)
    if len(target_free_paths) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("target-free path row count mismatch")
    artifacts = runtime_artifacts_dir()
    path_artifact = write_gzip_csv(
        target_free_paths, artifacts / f"{OUTPUT_PREFIX}_target_free_paths.csv.gz"
    )
    score_artifact = write_gzip_csv(
        target_free_scores,
        artifacts / f"{OUTPUT_PREFIX}_target_free_shift_scores.csv.gz",
    )
    target_free_contract = build_target_free_contract(
        config, path_artifact, score_artifact
    )
    contract_path = artifacts / f"{OUTPUT_PREFIX}_target_free_contract.json"
    write_json(contract_path, target_free_contract)

    # Suffix truth is first loaded here, after both target-free tables are persisted and hashed.
    truth = load_truth_after_freeze(
        exp226_path,
        config,
        target_free_contract_sha256=str(
            target_free_contract["target_free_contract_sha256"]
        ),
    )
    joined = attach_truth_after_freeze(
        target_free_paths,
        truth,
        target_free_contract_sha256=str(
            target_free_contract["target_free_contract_sha256"]
        ),
    )
    row_identity_coverage = float(len(joined) / len(target_free_paths))
    well_identity_coverage = float(
        joined["well_id"].nunique() / target_free_paths["well_id"].nunique()
    )
    finite_prediction_coverage = float(
        np.isfinite(joined["tvt_z"].to_numpy(np.float64)).mean()
    )
    stage_a_blocks = build_stage_a_block_readout(joined, config)
    stage_a_metrics = build_stage_a_metrics(stage_a_blocks)
    stage_a_gate = evaluate_stage_a_gate(
        stage_a_metrics,
        row_identity_coverage=row_identity_coverage,
        well_identity_coverage=well_identity_coverage,
        finite_prediction_coverage=finite_prediction_coverage,
        config=config,
    )
    hidden_assignments, hidden_manifest = load_hidden_like_assignments(config)
    stage_b_readout = build_stage_b_block_readout(target_free_scores, joined, config)
    stage_b_scope, stage_b_folds, stage_b_shifts = build_stage_b_metrics(
        stage_b_readout, hidden_assignments, config
    )
    stage_b_gate = evaluate_stage_b_gate(
        target_free_scores,
        stage_b_readout,
        stage_b_scope,
        stage_b_folds,
        config,
    )
    well_manifest = pd.DataFrame(well_manifest_rows).sort_values(
        "well_id", kind="mergesort"
    )
    input_manifest = pd.DataFrame(
        [
            exp226_manifest,
            hidden_manifest,
            {
                "name": "raw_train_well_files",
                "path": str(raw_dir),
                "rows": int(well_manifest["horizontal_rows"].sum()),
                "wells": len(well_manifest),
                "raw_sha256": frame_content_sha256(
                    well_manifest,
                    ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
                ),
            },
        ]
    )
    stage_a_block_artifact = write_gzip_csv(
        stage_a_blocks, artifacts / f"{OUTPUT_PREFIX}_stage_a_block_metrics.csv.gz"
    )
    stage_b_readout_artifact = write_gzip_csv(
        stage_b_readout, artifacts / f"{OUTPUT_PREFIX}_stage_b_block_readout.csv.gz"
    )
    plain_frames = {
        "stage_a_metrics": stage_a_metrics,
        "stage_b_scope_metrics": stage_b_scope,
        "stage_b_fold_metrics": stage_b_folds,
        "stage_b_shift_metrics": stage_b_shifts,
        "well_manifest": well_manifest,
        "input_manifest": input_manifest,
    }
    plain_paths: dict[str, Path] = {}
    for name, frame in plain_frames.items():
        path = artifacts / f"{OUTPUT_PREFIX}_{name}.csv"
        frame.to_csv(path, index=False)
        plain_paths[name] = path
    both_pass = bool(stage_a_gate["passed"] and stage_b_gate["passed"])
    decision_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_ab",
        "target_free_contract_sha256": target_free_contract[
            "target_free_contract_sha256"
        ],
        "stage_a": stage_a_gate,
        "stage_b": stage_b_gate,
        "stage_a_and_b_pass": both_pass,
        "stage_c_status": "eligible_for_separate_implementation_approval"
        if both_pass
        else "blocked_by_stage_ab_gate",
        "no_parameter_rescue": True,
    }
    decision_manifest["decision_manifest_sha256"] = json_sha256(decision_manifest)
    decision_path = artifacts / f"{OUTPUT_PREFIX}_decision_manifest.json"
    write_json(decision_path, decision_manifest)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage_ab_completed_both_pass"
        if both_pass
        else "stage_ab_completed_gate_failed",
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "rows": len(joined),
        "wells": int(joined["well_id"].nunique()),
        "folds": sorted(int(value) for value in joined["fold"].unique()),
        "stage_a": stage_a_gate,
        "stage_b": stage_b_gate,
        "execution": {
            "active_variants": 1,
            "diagnostic_contracts": 1,
            "fold_strata": 5,
            "model_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_well_runs": 0,
            "window_decoder_well_runs": 0,
            "parent_control_retraining": False,
        },
        "truth_attachment": {
            "stage": "after_target_free_path_and_score_freeze",
            "target_free_contract_sha256": target_free_contract[
                "target_free_contract_sha256"
            ],
        },
        "artifacts": {
            "target_free_path": path_artifact,
            "target_free_scores": score_artifact,
            "stage_a_blocks": stage_a_block_artifact,
            "stage_b_readout": stage_b_readout_artifact,
            "plain_file_sha256": {
                name: sha256_file(path) for name, path in plain_paths.items()
            },
            "target_free_contract": str(contract_path),
            "decision_manifest": str(decision_path),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": "request_separate_stage_c_implementation_approval"
        if both_pass
        else "close_stage_c_branch_without_rescue",
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": get_nested(config, "experiment.route"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "stage_a": stage_a_gate,
        "stage_b": stage_b_gate,
        "stage_c": None,
        "target_free_contract_sha256": target_free_contract[
            "target_free_contract_sha256"
        ],
        "notes": "Stage C, inference, prediction submission, and all model training remain disabled.",
    }
    write_json(runtime_metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Setup, contract preview, and execution


# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG, require_kaggle_approval=False)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "implementation_scope": get_nested(CONFIG, "implementation.scope"),
                "stage_a_horizons": get_nested(
                    CONFIG, "stage_a_residual_structure.block_horizons_rows"
                ),
                "stage_b_shifts": get_nested(
                    CONFIG, "stage_b_shift_separability.shift_bank_ft"
                ),
                "stage_c_enabled": get_nested(
                    CONFIG, "stage_c_window_gr_correction.enabled"
                ),
                "active_variants": get_nested(
                    CONFIG, "execution_contract.stage_ab.active_variants"
                ),
                "fold_strata": get_nested(
                    CONFIG, "execution_contract.stage_ab.fold_strata"
                ),
                "model_configs": get_nested(
                    CONFIG, "execution_contract.stage_ab.model_configs"
                ),
                "trained_folds": get_nested(
                    CONFIG, "execution_contract.stage_ab.trained_folds"
                ),
                "boosters": get_nested(CONFIG, "execution_contract.stage_ab.boosters"),
                "parent_control_retraining": get_nested(
                    CONFIG, "execution_contract.parent_control_retraining"
                ),
                "kaggle_push_approved": get_nested(
                    CONFIG, "execution_contract.kaggle_push_approved"
                ),
                "inference": get_nested(CONFIG, "execution_contract.inference"),
                "submission": get_nested(CONFIG, "execution_contract.submission"),
            },
            indent=2,
        )
    )


# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP321_STAGE_AB_SUMMARY = run_stage_ab(CONFIG)

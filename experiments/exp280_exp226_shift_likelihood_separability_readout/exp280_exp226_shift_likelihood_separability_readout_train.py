# %% [markdown]
# # exp280 exp226 shift-likelihood separability readout
#
# This is a target-free, zero-booster diagnostic. It freezes raw-GR/typewell
# likelihood scores for a fixed vertical-shift bank around group-safe exp226
# `tvt_geop`, hashes that score table, and only then attaches true TVT for
# separability readouts. It never creates a corrected prediction or submission.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Exp226 cache and raw-well input checks
# 4. Fixed Gaussian shift-likelihood scoring
# 5. Truth-only block labels and persistent-offset readout
# 6. Fold, scope, shift, and well metrics
# 7. Full Kaggle CPU orchestration and artifact guards
# 8. Setup and contract preview
# 9. Run diagnostic and report generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_NAME = "exp280_exp226_shift_likelihood_separability_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP280_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    raise FileNotFoundError(f"exp280 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: dict[str, Any]) -> Path:
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


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": dataframe_content_sha(frame),
    }


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


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


def validate_scientific_contract(config: dict[str, Any]) -> None:
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
    shifts = [float(value) for value in get_nested(config, "audit.shift_bank_ft") or []]
    emission = get_nested(config, "audit.emission") or {}
    guards = get_nested(config, "validation.guards") or {}
    if shifts != expected_shifts:
        raise ValueError("exp280 fixes the approved 13-value shift bank")
    if int(get_nested(config, "audit.block_rows") or 0) != 512:
        raise ValueError("exp280 fixes non-overlapping 512-row blocks")
    if (
        get_nested(config, "audit.block_policy")
        != "non_overlapping_from_suffix_start_keep_short_tail"
    ):
        raise ValueError("exp280 fixes the non-overlapping short-tail block policy")
    if get_nested(config, "audit.score_aggregation") != "mean_row_log_likelihood":
        raise ValueError("exp280 fixes mean row log-likelihood aggregation")
    if get_nested(config, "audit.tie_policy") != "config_shift_bank_order":
        raise ValueError("exp280 fixes config-order tie resolution")
    if emission.get("kind") != "exp209_gaussian_raw_gr":
        raise ValueError("exp280 fixes the exp209 Gaussian raw-GR emission")
    if [float(value) for value in emission.get("sigma_clip", [])] != [10.0, 60.0]:
        raise ValueError("exp280 fixes GR sigma clip [10, 60]")
    if float(emission.get("log_likelihood_clip", 0.0)) != 600.0:
        raise ValueError("exp280 fixes Gaussian log-likelihood clip 600")
    for key in (
        "minimum_folds_real_top1_above_shuffled",
        "minimum_folds_real_top3_above_shuffled",
        "minimum_folds_real_mrr_above_shuffled",
        "minimum_folds_real_sign_above_shuffled",
    ):
        if int(guards.get(key, 0)) != 5:
            raise ValueError("exp280 fixes all four separability guards at 5/5 folds")
    expected_zero = {
        "model.lightgbm_config_count": 0,
        "model.trained_fold_count": 0,
        "model.booster_count": 0,
        "model.hmm_decode_count": 0,
        "execution.total_boosters": 0,
        "execution.hmm_well_runs": 0,
    }
    for key, expected in expected_zero.items():
        if int(get_nested(config, key) or 0) != expected:
            raise ValueError(f"exp280 requires {key}={expected}")
    if bool(get_nested(config, "execution.inference")) or bool(
        get_nested(config, "execution.submission")
    ):
        raise ValueError("exp280 forbids inference and submission")


# %% [markdown]
# ## 3. Exp226 cache and raw-well input checks


# %%
def load_exp226_safe(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof") or {}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_decompressed_sha = sha256_gzip_decompressed(path)
    expected_decompressed_sha = str(spec["expected_decompressed_sha256"])
    if actual_decompressed_sha != expected_decompressed_sha:
        raise ValueError(
            "exp226 decompressed SHA mismatch: "
            f"{actual_decompressed_sha} != {expected_decompressed_sha}"
        )
    safe_columns = [str(value) for value in spec["safe_columns"]]
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF has duplicate well_id/row_idx")
    if not np.isfinite(frame["tvt_geop"]).all():
        raise ValueError("exp226 tvt_geop must be finite")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if len(frame) != expected_rows or frame["well_id"].nunique() != expected_wells:
        raise ValueError("exp226 row/well coverage does not match the fixed contract")
    if sorted(frame["fold"].unique().tolist()) != expected_folds:
        raise ValueError("exp226 fold set does not match the fixed contract")
    fold_counts = frame.groupby("well_id")["fold"].nunique()
    if not bool((fold_counts == 1).all()):
        raise ValueError("each exp226 well must belong to exactly one fold")
    manifest = {
        "name": "exp226_group_safe_oof_safe_columns",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": actual_decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
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
        raise ValueError("truth attachment requires a frozen target-free score content SHA")
    spec = get_nested(config, "data.exp226_oof") or {}
    truth_columns = [str(value) for value in spec["truth_columns"]]
    frame = pd.read_csv(path, usecols=truth_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any() or not np.isfinite(frame["tvt_true"]).all():
        raise ValueError("exp226 truth readout rows must be unique and finite")
    return frame


def load_hidden_like_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    if not bool(spec.get("enabled")):
        return pd.DataFrame(), {"enabled": False}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *[str(value) for value in spec["role_columns"].values()]}
    if not required.issubset(frame.columns):
        raise ValueError(f"hidden-like assignments missing {sorted(required - set(frame.columns))}")
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments require one row per well")
    manifest = {
        "name": "exp115_hidden_like_fold_assignments",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }
    return frame, manifest


def load_horizontal_without_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda column: column != "TVT")
    if "TVT" in frame.columns:
        raise ValueError("target-free horizontal reader must not expose TVT")
    return frame


# %% [markdown]
# ## 4. Fixed Gaussian shift-likelihood scoring


# %%
def prepare_gr_inputs(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free GR preparation forbids horizontal TVT")
    required_horizontal = {"MD", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal_without_truth.columns):
        missing = sorted(required_horizontal - set(horizontal_without_truth.columns))
        raise ValueError(
            f"horizontal missing {missing}"
        )
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
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
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "audit.emission.sigma_clip")
    ]
    gr_sigma = float(np.clip(np.nanstd(residual), sigma_low, sigma_high))
    if not np.isfinite(gr_sigma):
        raise ValueError("known-prefix GR residual sigma is not finite")
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
        "gr_sigma": gr_sigma,
        "all_gr_interpolated": all_gr,
        "known_rows": len(known),
        "known_residual_mean": float(np.mean(residual)),
        "known_residual_std_unclipped": float(np.std(residual)),
    }


def score_well_target_free(
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
        raise ValueError(f"target-free score input contains forbidden exp226 columns: {leaked}")
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free score input contains horizontal TVT")
    required_oof = {"well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"}
    if not required_oof.issubset(oof_safe.columns):
        raise ValueError(f"safe OOF missing {sorted(required_oof - set(oof_safe.columns))}")
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if oof.empty or oof["well_id"].nunique() != 1 or oof["fold"].nunique() != 1:
        raise ValueError("score_well_target_free requires one non-empty well and fold")
    row_idx = oof["row_idx"].to_numpy(np.int64)
    suffix_offset = oof["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(oof), dtype=np.int64)):
        raise ValueError("exp226 suffix_offset must be contiguous from zero")
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal_without_truth):
        raise ValueError("exp226 row_idx is outside the raw horizontal frame")
    if horizontal_without_truth.iloc[row_idx]["TVT_input"].notna().any():
        raise ValueError("exp226 OOF rows must align only to unknown-suffix rows")

    prepared = prepare_gr_inputs(horizontal_without_truth, typewell, config)
    shifts = np.asarray(get_nested(config, "audit.shift_bank_ft"), dtype=np.float64)
    block_rows = int(get_nested(config, "audit.block_rows"))
    geop = oof["tvt_geop"].to_numpy(np.float64)
    candidate_tvt = geop[:, None] + shifts[None, :]
    expected_gr = np.empty_like(candidate_tvt)
    for slot in range(len(shifts)):
        expected_gr[:, slot] = np.interp(
            candidate_tvt[:, slot], prepared["typewell_tvt"], prepared["typewell_gr"]
        )
    raw_gr = prepared["all_gr_interpolated"][row_idx]
    clip_value = float(get_nested(config, "audit.emission.log_likelihood_clip"))
    zscore = (raw_gr[:, None] - expected_gr) / float(prepared["gr_sigma"])
    log_likelihood = -0.5 * np.minimum(zscore**2, clip_value)
    if not np.isfinite(log_likelihood).all():
        raise ValueError("target-free shift likelihood must be finite")

    observed_gr = pd.to_numeric(horizontal_without_truth.iloc[row_idx]["GR"], errors="coerce")
    md = pd.to_numeric(horizontal_without_truth["MD"], errors="raise").to_numpy(np.float64)
    known_positions = np.flatnonzero(horizontal_without_truth["TVT_input"].notna().to_numpy())
    if not len(known_positions):
        raise ValueError("well has no known TVT_input prefix")
    last_known = int(known_positions[-1])
    md_since = md[row_idx] - md[last_known]
    block_id = suffix_offset // block_rows
    native = (candidate_tvt >= prepared["typewell_tvt"].min()) & (
        candidate_tvt <= prepared["typewell_tvt"].max()
    )
    extension = float(get_nested(config, "audit.typewell_extension_ft"))
    extended = (candidate_tvt >= prepared["typewell_tvt"].min() - extension) & (
        candidate_tvt <= prepared["typewell_tvt"].max() + extension
    )

    well = str(oof["well_id"].iloc[0])
    fold = int(oof["fold"].iloc[0])
    shuffle_seed = int(get_nested(config, "audit.shuffled_control.seed"))
    rows: list[dict[str, Any]] = []
    for block in np.unique(block_id):
        mask = block_id == block
        scores = log_likelihood[mask].mean(axis=0)
        score_sums = log_likelihood[mask].sum(axis=0)
        ranks = rank_descending(scores)
        rng = np.random.default_rng(stable_seed(EXPERIMENT_NAME, shuffle_seed, well, int(block)))
        shuffled_scores = scores[rng.permutation(len(scores))]
        shuffled_ranks = rank_descending(shuffled_scores)
        block_positions = np.flatnonzero(mask)
        for slot, shift in enumerate(shifts):
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "block_id": int(block),
                    "block_start_suffix_offset": int(suffix_offset[block_positions[0]]),
                    "block_end_suffix_offset": int(suffix_offset[block_positions[-1]]),
                    "block_start_row_idx": int(row_idx[block_positions[0]]),
                    "block_end_row_idx": int(row_idx[block_positions[-1]]),
                    "block_row_count": int(mask.sum()),
                    "md_since_min_ft": float(np.min(md_since[mask])),
                    "md_since_max_ft": float(np.max(md_since[mask])),
                    "md_since_mid_ft": float(np.mean(md_since[mask])),
                    "observed_gr_share": float(observed_gr.iloc[block_positions].notna().mean()),
                    "shift_slot": int(slot),
                    "shift_ft": float(shift),
                    "likelihood_mean": float(scores[slot]),
                    "likelihood_sum": float(score_sums[slot]),
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
        "well_id": well,
        "fold": fold,
        "horizontal_rows": len(horizontal_without_truth),
        "evaluation_rows": len(oof),
        "blocks": int(block_id.max() + 1),
        "known_rows": int(prepared["known_rows"]),
        "last_known_row_idx": last_known,
        "gr_sigma": float(prepared["gr_sigma"]),
        "known_residual_mean": float(prepared["known_residual_mean"]),
        "known_residual_std_unclipped": float(prepared["known_residual_std_unclipped"]),
        "observed_eval_gr_share": float(observed_gr.notna().mean()),
        "score_finite_coverage": float(np.isfinite(log_likelihood).mean()),
    }
    return score_frame.reset_index(drop=True), manifest


# %% [markdown]
# ## 5. Truth-only block labels and persistent-offset readout


# %%
def persistent_offset_episodes(
    signed_base_error: np.ndarray,
    row_idx: np.ndarray,
    *,
    threshold_ft: float,
    minimum_consecutive_rows: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    error = np.asarray(signed_base_error, dtype=np.float64)
    indices = np.asarray(row_idx, dtype=np.int64)
    if error.ndim != 1 or len(error) != len(indices):
        raise ValueError("persistent-offset inputs must be aligned one-dimensional arrays")
    bad = np.abs(error) > float(threshold_ft)
    padded = np.concatenate([[False], bad, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    mask = np.zeros(len(error), dtype=bool)
    episodes: list[dict[str, Any]] = []
    for start, end in zip(starts, ends, strict=True):
        if end - start < int(minimum_consecutive_rows):
            continue
        mask[start:end] = True
        segment = error[start:end]
        episodes.append(
            {
                "episode_start_row_idx": int(indices[start]),
                "episode_end_row_idx": int(indices[end - 1]),
                "episode_rows": int(end - start),
                "median_signed_base_error_ft": float(np.median(segment)),
                "peak_abs_base_error_ft": float(np.max(np.abs(segment))),
            }
        )
    return mask, episodes


def sign_match(selected_shift: float, nearest_shift: float) -> bool:
    return bool(np.sign(float(selected_shift)) == np.sign(float(nearest_shift)))


def build_truth_readout(
    target_free_scores: pd.DataFrame,
    oof_safe: pd.DataFrame,
    truth: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(truth) != len(oof_safe):
        raise ValueError("truth and safe OOF row counts must match before attachment")
    merged = oof_safe.merge(truth, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    if len(merged) != len(oof_safe) or merged["tvt_true"].isna().any():
        raise ValueError("truth attachment failed row identity coverage")
    merged = merged.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    shifts = np.asarray(get_nested(config, "audit.shift_bank_ft"), dtype=np.float64)
    block_rows = int(get_nested(config, "audit.block_rows"))
    persistent_spec = get_nested(config, "audit.persistent_offset") or {}
    maximum_quantization_error = float(
        get_nested(config, "audit.coverage.maximum_quantization_error_ft")
    )
    readout_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for well, well_frame in merged.groupby("well_id", sort=True):
        well_frame = well_frame.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        fold = int(well_frame["fold"].iloc[0])
        row_index = well_frame["row_idx"].to_numpy(np.int64)
        base_error = well_frame["tvt_geop"].to_numpy(np.float64) - well_frame[
            "tvt_true"
        ].to_numpy(np.float64)
        persistent_mask, well_episodes = persistent_offset_episodes(
            base_error,
            row_index,
            threshold_ft=float(persistent_spec["error_threshold_ft"]),
            minimum_consecutive_rows=int(persistent_spec["minimum_consecutive_rows"]),
        )
        for episode_id, row in enumerate(well_episodes):
            episode_rows.append(
                {"well_id": str(well), "fold": fold, "episode_id": episode_id, **row}
            )
        block_id = well_frame["suffix_offset"].to_numpy(np.int64) // block_rows
        for block in np.unique(block_id):
            mask = block_id == block
            block_frame = well_frame.loc[mask]
            block_scores = target_free_scores.loc[
                (target_free_scores["well_id"].astype(str) == str(well))
                & (target_free_scores["block_id"] == int(block))
            ].sort_values("shift_slot", kind="mergesort")
            if len(block_scores) != len(shifts) or not np.array_equal(
                block_scores["shift_ft"].to_numpy(np.float64), shifts
            ):
                raise ValueError(f"target-free score bank misalignment for {well} block {block}")
            true_tvt = block_frame["tvt_true"].to_numpy(np.float64)
            geop = block_frame["tvt_geop"].to_numpy(np.float64)
            errors = geop[:, None] + shifts[None, :] - true_tvt[:, None]
            candidate_rmse = np.sqrt(np.mean(errors**2, axis=0))
            nearest_slot = int(np.argmin(candidate_rmse))
            real_rank = int(block_scores["likelihood_rank"].iloc[nearest_slot])
            shuffled_rank = int(
                block_scores["shuffled_likelihood_rank"].iloc[nearest_slot]
            )
            top1_slot = int(np.argmin(block_scores["likelihood_rank"].to_numpy(np.int64)))
            shuffled_top1_slot = int(
                np.argmin(block_scores["shuffled_likelihood_rank"].to_numpy(np.int64))
            )
            likelihood = block_scores["likelihood_mean"].to_numpy(np.float64)
            ordered_likelihood = np.sort(likelihood)[::-1]
            other = np.delete(likelihood, nearest_slot)
            continuous_optimal_shift = float(np.mean(true_tvt - geop))
            nearest_shift = float(shifts[nearest_slot])
            top1_shift = float(shifts[top1_slot])
            shuffled_top1_shift = float(shifts[shuffled_top1_slot])
            top1_rmse = float(candidate_rmse[top1_slot])
            nearest_rmse = float(candidate_rmse[nearest_slot])
            base_rmse = float(np.sqrt(np.mean((geop - true_tvt) ** 2)))
            base_row_positions = np.flatnonzero(mask)
            score_meta = block_scores.iloc[0]
            readout_rows.append(
                {
                    "well_id": str(well),
                    "fold": fold,
                    "block_id": int(block),
                    "block_start_row_idx": int(block_frame["row_idx"].iloc[0]),
                    "block_end_row_idx": int(block_frame["row_idx"].iloc[-1]),
                    "block_row_count": len(block_frame),
                    "md_since_min_ft": float(score_meta["md_since_min_ft"]),
                    "md_since_max_ft": float(score_meta["md_since_max_ft"]),
                    "md_since_mid_ft": float(score_meta["md_since_mid_ft"]),
                    "observed_gr_share": float(score_meta["observed_gr_share"]),
                    "continuous_optimal_shift_ft": continuous_optimal_shift,
                    "nearest_shift_ft": nearest_shift,
                    "nearest_shift_slot": nearest_slot,
                    "nearest_shift_rank": real_rank,
                    "nearest_shift_shuffled_rank": shuffled_rank,
                    "top1_hit": bool(real_rank == 1),
                    "top3_hit": bool(real_rank <= 3),
                    "mrr": float(1.0 / real_rank),
                    "normalized_rank": float((real_rank - 1) / max(len(shifts) - 1, 1)),
                    "shuffled_top1_hit": bool(shuffled_rank == 1),
                    "shuffled_top3_hit": bool(shuffled_rank <= 3),
                    "shuffled_mrr": float(1.0 / shuffled_rank),
                    "shuffled_normalized_rank": float(
                        (shuffled_rank - 1) / max(len(shifts) - 1, 1)
                    ),
                    "top1_shift_ft": top1_shift,
                    "shuffled_top1_shift_ft": shuffled_top1_shift,
                    "sign_match": sign_match(top1_shift, nearest_shift),
                    "shuffled_sign_match": sign_match(shuffled_top1_shift, nearest_shift),
                    "likelihood_top1_margin": float(
                        ordered_likelihood[0] - ordered_likelihood[1]
                    ),
                    "truth_candidate_margin": float(likelihood[nearest_slot] - np.max(other)),
                    "base_rmse": base_rmse,
                    "nearest_shift_rmse": nearest_rmse,
                    "top1_shift_rmse": top1_rmse,
                    "top1_regret_rmse": float(top1_rmse - nearest_rmse),
                    "oracle_shift_gain_rmse": float(base_rmse - nearest_rmse),
                    "bank_range_covered": bool(
                        shifts.min() <= continuous_optimal_shift <= shifts.max()
                    ),
                    "nearest_shift_quantization_error_ft": float(
                        abs(nearest_shift - continuous_optimal_shift)
                    ),
                    "quantization_covered": bool(
                        abs(nearest_shift - continuous_optimal_shift)
                        <= maximum_quantization_error
                    ),
                    "persistent_offset_share": float(persistent_mask[base_row_positions].mean()),
                    "persistent_offset_block": bool(persistent_mask[base_row_positions].any()),
                }
            )
    readout = pd.DataFrame(readout_rows).sort_values(
        ["well_id", "block_id"], kind="mergesort"
    )
    episodes = pd.DataFrame(episode_rows)
    return readout.reset_index(drop=True), episodes.reset_index(drop=True)


# %% [markdown]
# ## 6. Fold, scope, shift, and well metrics


# %%
def readout_metric_row(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"scope {scope} selected zero blocks")
    return {
        "scope": scope,
        "blocks": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "top1_rate": float(frame["top1_hit"].mean()),
        "top3_rate": float(frame["top3_hit"].mean()),
        "mrr": float(frame["mrr"].mean()),
        "mean_rank": float(frame["nearest_shift_rank"].mean()),
        "mean_normalized_rank": float(frame["normalized_rank"].mean()),
        "sign_accuracy": float(frame["sign_match"].mean()),
        "shuffled_top1_rate": float(frame["shuffled_top1_hit"].mean()),
        "shuffled_top3_rate": float(frame["shuffled_top3_hit"].mean()),
        "shuffled_mrr": float(frame["shuffled_mrr"].mean()),
        "shuffled_mean_rank": float(frame["nearest_shift_shuffled_rank"].mean()),
        "shuffled_mean_normalized_rank": float(frame["shuffled_normalized_rank"].mean()),
        "shuffled_sign_accuracy": float(frame["shuffled_sign_match"].mean()),
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
        "oracle_shift_gain_rmse_mean": float(frame["oracle_shift_gain_rmse"].mean()),
        "top1_regret_rmse_mean": float(frame["top1_regret_rmse"].mean()),
        "top1_regret_rmse_p90": float(frame["top1_regret_rmse"].quantile(0.90)),
        "likelihood_top1_margin_mean": float(frame["likelihood_top1_margin"].mean()),
        "truth_candidate_margin_mean": float(frame["truth_candidate_margin"].mean()),
    }


def build_scope_metrics(
    readout: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_rows = [readout_metric_row(readout, scope="overall")]
    near_limit = float(get_nested(config, "audit.scopes.near_max_md_since_ft"))
    long_limit = float(get_nested(config, "audit.scopes.long_tail_min_md_since_ft"))
    predefined = {
        "near": readout["md_since_mid_ft"] < near_limit,
        "long_tail_1000_plus": readout["md_since_mid_ft"] >= long_limit,
        "persistent_offset": readout["persistent_offset_block"].astype(bool),
    }
    for name, mask in predefined.items():
        if bool(mask.any()):
            scope_rows.append(readout_metric_row(readout.loc[mask], scope=name))
    if not hidden_assignments.empty:
        role_columns = get_nested(config, "data.hidden_like.role_columns") or {}
        role_by_well = hidden_assignments.set_index("well_id")
        for scope_name, role_column in role_columns.items():
            valid_wells = set(
                role_by_well.index[
                    role_by_well[str(role_column)].astype(str) == "valid"
                ].astype(str)
            )
            part = readout.loc[readout["well_id"].astype(str).isin(valid_wells)]
            scope_rows.append(readout_metric_row(part, scope=str(scope_name)))
    fold_rows = []
    for fold, part in readout.groupby("fold", sort=True):
        row = readout_metric_row(part, scope=f"fold_{int(fold)}")
        row["fold"] = int(fold)
        fold_rows.append(row)
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def build_shift_metrics(readout: pd.DataFrame, shifts: list[float]) -> pd.DataFrame:
    rows = []
    for shift in shifts:
        nearest = readout.loc[np.isclose(readout["nearest_shift_ft"], float(shift))]
        selected = readout.loc[np.isclose(readout["top1_shift_ft"], float(shift))]
        rows.append(
            {
                "shift_ft": float(shift),
                "truth_nearest_blocks": len(nearest),
                "truth_nearest_share": float(len(nearest) / len(readout)),
                "likelihood_top1_blocks": len(selected),
                "likelihood_top1_share": float(len(selected) / len(readout)),
                "top1_rate_when_truth_nearest": float(nearest["top1_hit"].mean())
                if len(nearest)
                else np.nan,
                "top3_rate_when_truth_nearest": float(nearest["top3_hit"].mean())
                if len(nearest)
                else np.nan,
                "mean_rank_when_truth_nearest": float(nearest["nearest_shift_rank"].mean())
                if len(nearest)
                else np.nan,
                "mean_quantization_error_ft": float(
                    nearest["nearest_shift_quantization_error_ft"].mean()
                )
                if len(nearest)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_by_well_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, part in readout.groupby("well_id", sort=True):
        row = readout_metric_row(part, scope=str(well))
        row["well_id"] = str(well)
        row["fold"] = int(part["fold"].iloc[0])
        row["persistent_offset_blocks"] = int(part["persistent_offset_block"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_guard(
    target_free_scores: pd.DataFrame,
    readout: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.guards") or {}
    expected_folds = [int(value) for value in guards["required_folds"]]
    actual_folds = sorted(int(value) for value in fold_metrics["fold"].unique())
    row_identity_coverage = 1.0  # build_truth_readout hard-fails any missing or duplicate join.
    finite_coverage = float(
        np.isfinite(
            target_free_scores[["likelihood_mean", "shuffled_likelihood_mean"]].to_numpy(
                np.float64
            )
        ).mean()
    )
    comparisons = {
        "top1": fold_metrics["top1_rate"] > fold_metrics["shuffled_top1_rate"],
        "top3": fold_metrics["top3_rate"] > fold_metrics["shuffled_top3_rate"],
        "mrr": fold_metrics["mrr"] > fold_metrics["shuffled_mrr"],
        "sign": fold_metrics["sign_accuracy"] > fold_metrics["shuffled_sign_accuracy"],
    }
    counts = {name: int(values.sum()) for name, values in comparisons.items()}
    required_counts = {
        "top1": int(guards["minimum_folds_real_top1_above_shuffled"]),
        "top3": int(guards["minimum_folds_real_top3_above_shuffled"]),
        "mrr": int(guards["minimum_folds_real_mrr_above_shuffled"]),
        "sign": int(guards["minimum_folds_real_sign_above_shuffled"]),
    }
    checks = {
        "expected_folds": actual_folds == expected_folds,
        "score_finite_coverage": finite_coverage
        >= float(guards["required_score_finite_coverage"]),
        "row_identity_coverage": row_identity_coverage
        >= float(guards["required_row_identity_coverage"]),
        **{
            f"{name}_all_folds_above_shuffled": counts[name] >= required_counts[name]
            for name in comparisons
        },
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "actual_folds": actual_folds,
        "score_finite_coverage": finite_coverage,
        "row_identity_coverage": row_identity_coverage,
        "block_readout_rows": len(readout),
        "folds_real_above_shuffled": counts,
        "required_folds_real_above_shuffled": required_counts,
    }


# %% [markdown]
# ## 7. Full Kaggle CPU orchestration and artifact guards


# %%
def run_full_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp280 readout must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is reserved "
            "for an explicitly approved local smoke run."
        )
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("exp280 Kaggle CPU execution is not approved")
    validate_scientific_contract(config)
    started = time.time()
    safe_oof, exp226_path, exp226_manifest = load_exp226_safe(config)
    raw_dir = train_data_dir(config)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    raw_wells = sorted(
        path.name.replace("__horizontal_well.csv", "")
        for path in raw_dir.glob("*__horizontal_well.csv")
    )
    if len(raw_wells) != expected_wells or set(raw_wells) != set(safe_oof["well_id"].unique()):
        raise ValueError("raw train and exp226 well sets do not match")

    score_parts: list[pd.DataFrame] = []
    well_manifest_rows: list[dict[str, Any]] = []
    progress_every = 25
    for index, well in enumerate(raw_wells, start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        horizontal_safe = load_horizontal_without_truth(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        well_scores, well_manifest = score_well_target_free(
            safe_oof.loc[safe_oof["well_id"] == well], horizontal_safe, typewell, config
        )
        well_manifest.update(
            {
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
        score_parts.append(well_scores)
        well_manifest_rows.append(well_manifest)
        if index % progress_every == 0 or index == len(raw_wells):
            print(f"target-free scoring wells={index}/{len(raw_wells)}")

    target_free_scores = pd.concat(score_parts, ignore_index=True).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    target_free_score_content_sha = dataframe_content_sha(target_free_scores)
    if not target_free_score_content_sha:
        raise RuntimeError("failed to freeze target-free score content SHA")
    artifacts = artifact_dir()
    score_contract = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "shift_bank_ft": get_nested(config, "audit.shift_bank_ft"),
        "block_rows": get_nested(config, "audit.block_rows"),
        "block_policy": get_nested(config, "audit.block_policy"),
        "score_aggregation": get_nested(config, "audit.score_aggregation"),
        "tie_policy": get_nested(config, "audit.tie_policy"),
        "emission": get_nested(config, "audit.emission"),
        "shuffled_control": get_nested(config, "audit.shuffled_control"),
        "target_free_score_content_sha256": target_free_score_content_sha,
    }
    score_contract["scientific_contract_sha256"] = mapping_sha256(score_contract)
    score_contract_path = artifacts / f"{OUTPUT_PREFIX}_score_contract.json"
    write_json(score_contract_path, score_contract)
    score_artifact = write_csv_gzip(
        target_free_scores,
        artifacts / f"{OUTPUT_PREFIX}_target_free_shift_scores.csv.gz",
    )

    # Truth is first read here, after every target-free score is frozen and hashed.
    truth = load_exp226_truth(
        exp226_path,
        config,
        frozen_score_content_sha256=target_free_score_content_sha,
    )
    readout, episodes = build_truth_readout(target_free_scores, safe_oof, truth, config)
    hidden_assignments, hidden_manifest = load_hidden_like_assignments(config)
    scope_metrics, fold_metrics = build_scope_metrics(readout, hidden_assignments, config)
    shift_metrics = build_shift_metrics(
        readout, [float(value) for value in get_nested(config, "audit.shift_bank_ft")]
    )
    by_well = build_by_well_metrics(readout)
    well_manifest = pd.DataFrame(well_manifest_rows).sort_values("well_id", kind="mergesort")
    guard = evaluate_guard(target_free_scores, readout, fold_metrics, config)

    readout_artifact = write_csv_gzip(
        readout,
        artifacts / f"{OUTPUT_PREFIX}_block_readout.csv.gz",
    )
    file_frames = {
        "scope_metrics": scope_metrics,
        "fold_metrics": fold_metrics,
        "shift_metrics": shift_metrics,
        "by_well_metrics": by_well,
        "persistent_offset_episodes": episodes,
        "well_manifest": well_manifest,
    }
    output_paths: dict[str, Path] = {}
    for name, frame in file_frames.items():
        path = artifacts / f"{OUTPUT_PREFIX}_{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = path

    input_manifest = pd.DataFrame(
        [
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

    overall = scope_metrics.loc[scope_metrics["scope"] == "overall"].iloc[0].to_dict()
    hashed_outputs = {**output_paths, "input_manifest": input_manifest_path}
    output_sha = {name: sha256_path(path) for name, path in hashed_outputs.items()}
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_side_readout_completed_guard_passed"
        if guard["passed"]
        else "train_side_readout_completed_guard_failed",
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "rows": len(safe_oof),
        "wells": int(safe_oof["well_id"].nunique()),
        "blocks": len(readout),
        "shift_candidates": len(get_nested(config, "audit.shift_bank_ft")),
        "active_audit_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "overall": overall,
        "guard": guard,
        "truth_attachment": {
            "stage": "after_all_target_free_scores_frozen",
            "target_free_score_content_sha256": target_free_score_content_sha,
        },
        "input_manifest": input_manifest.to_dict(orient="records"),
        "artifacts": {
            "score_contract": str(score_contract_path),
            "target_free_scores": score_artifact,
            "block_readout": readout_artifact,
            "file_sha256": output_sha,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": "separability_supported_consider_separate_residual_offset_hmm"
        if guard["passed"]
        else "separability_not_fold_stable_lower_residual_offset_hmm_priority",
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "diagnostic": {
            "overall": overall,
            "guard": guard,
            "target_free_score_content_sha256": target_free_score_content_sha,
        },
        "notes": "No prediction, model, inference, or submission is produced.",
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 8. Setup and contract preview


# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "stage": get_nested(CONFIG, "execution.stage"),
                "shift_bank_ft": get_nested(CONFIG, "audit.shift_bank_ft"),
                "block_rows": get_nested(CONFIG, "audit.block_rows"),
                "active_audit_variants": get_nested(CONFIG, "execution.active_audit_variants"),
                "lightgbm_configs": get_nested(CONFIG, "execution.lightgbm_config_count"),
                "trained_folds": get_nested(CONFIG, "execution.trained_fold_count"),
                "boosters": get_nested(CONFIG, "execution.total_boosters"),
                "hmm_well_runs": get_nested(CONFIG, "execution.hmm_well_runs"),
                "inference": get_nested(CONFIG, "execution.inference"),
                "submission": get_nested(CONFIG, "execution.submission"),
            },
            indent=2,
        )
    )


# %% [markdown]
# ## 9. Run diagnostic and report generated artifacts


# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP280_SUMMARY = run_full_experiment(CONFIG)
